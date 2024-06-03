from gensim.utils import simple_preprocess
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import numpy as np
from scipy import spatial
from typing import Tuple, List, Optional
from datasets import load_dataset
import tensorflow as tf
from gensim.models import fasttext
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import pearsonr

class TextSimilarity:
    def __init__(self, model, dataset, remap_embeddings = None, mode = "mean", rob=True, pretrained = False, remap=True, trainable = True):
        self.input_pairs = [(e["sentence1"], e["sentence2"], e["label"], ) for e in dataset["train"].to_list()]
        self.input_pairs_val = [(e["sentence1"], e["sentence2"], e["label"], ) for e in dataset["validation"].to_list()]
        self.input_pairs_test = [(e["sentence1"], e["sentence2"], e["label"], ) for e in dataset["test"].to_list()]
        self.model = model
        self.create_dict_tfid()
        self.rob = rob
        self.pretrained= pretrained
        self.remap_embeddings = remap
        self.trainable = trainable
        self.mapped_train = self.map_pairs(self.input_pairs, dictionary=self.diccionari, mode = mode)
        self.mapped_val = self.map_pairs(self.input_pairs_val, dictionary=self.diccionari, mode=mode)
        self.mapped_test = self.map_pairs(self.input_pairs_test, dictionary=self.diccionari, mode=mode)
        self._pretrained_weights: Optional[np.ndarray] = None
        if self.pretrained:
            self.pretrained_weights()

    def __simple_preprocess(self, sentence:str) -> List[str]:
        preprocessed = simple_preprocess(sentence)
        return preprocessed
    
    def create_dict_tfid(self):
        all_input_pairs = self.input_pairs + self.input_pairs_val + self.input_pairs_test
        sentences_1_preproc = [simple_preprocess(sentence_1) for sentence_1, _, _ in all_input_pairs]
        sentences_2_preproc = [simple_preprocess(sentence_2) for _, sentence_2, _ in all_input_pairs]
        sentence_pairs = list(zip(sentences_1_preproc, sentences_2_preproc))
        # Versión aplanada para poder entrenar el modelo
        sentences_pairs_flattened = sentences_1_preproc + sentences_2_preproc
        self.diccionari = Dictionary(sentences_pairs_flattened)
        corpus = [self.diccionari.doc2bow(sent) for sent in sentences_pairs_flattened]
        self.model_tfidf = TfidfModel(corpus)

    def _map_tf_idf(self, sentence_preproc: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        bow = self.diccionari.doc2bow(sentence_preproc)
        tf_idf = self.model_tfidf[bow]
        vectors, weights = [], []
        for word_index, weight in tf_idf:
            word = self.diccionari.get(word_index)
            if word in self.model:
                vectors.append(self.model[word])
                weights.append(weight)
        return np.average(vectors, weights=weights, axis=0,)
    
    def _map_word_embeddings(self,
        sentence: str,
        sequence_len: int = 96,
        fixed_dictionary: Optional[Dictionary] = None
    ) -> np.ndarray:
        """
        Map to word-embedding indices
        :param sentence:
        :param sequence_len:
        :param fixed_dictionary:
        :return:
        """
        sentence_preproc = simple_preprocess(sentence)[:sequence_len]
        _vectors = np.zeros(sequence_len, dtype=np.int32)
        index = 0
        for word in sentence_preproc:
            if fixed_dictionary is not None:
                if word in fixed_dictionary.token2id:
                    # Sumo 1 porque el valor 0 está reservado a padding
                    _vectors[index] = fixed_dictionary.token2id[word] + 1
                    index += 1
            else:
                if word in self.model.key_to_index:
                    _vectors[index] = self.model.key_to_index[word] + 1
                    index += 1
        return _vectors
    
    def _map_mean(self, sentence):
        vector = [self.model[word] for word in sentence if word in self.model]
        return np.mean(vector, axis=0)
    
    def _map_spacy(self, sentence):
        self.model(sentence)
        return sentence.vector
    
    def _map_roberta(self, sentence, cls=True):
        doc = self.model(sentence)
        if cls:
            return doc._.trf_data.last_hidden_layer_state.data[-1]
        else:
            return np.mean(doc._.trf_data.last_hidden_layer_state.data[:-1], axis=0)
        
    def _map_one_hot(self, sentence):
        new_dict = self.diccionari
        new_dict.filter_extremes(no_below=20)
        vector = np.zeros(len(new_dict), dtype = np.float32)
        for word_index, _ in new_dict.doc2bow(sentence):
            vector[word_index] = 1
        return vector

    def map_pairs(self,
        sentence_pairs: List[Tuple[str, str, float]],
        dictionary: Dictionary = None,
        mode = None,
        tf_idf_model: TfidfModel = None,
        sequence_len: int = 96,
        fixed_dictionary: Optional[Dictionary] = None) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
        pares_vectores = []
        self.mode = mode
        self.sequence_len = sequence_len
        for i, (sentence_1, sentence_2, similitud) in enumerate(sentence_pairs):
            sentence_1_preproc = self.__simple_preprocess(sentence_1)
            sentence_2_preproc = self.__simple_preprocess(sentence_2)
            # Si usamos TF-IDF
            if mode == "tfidf":
                # Cálculo del promedio ponderado por TF-IDF de los word embeddings
                vector1 = self._map_tf_idf(sentence_1_preproc )
                vector2 = self._map_tf_idf(sentence_2_preproc)
            elif mode == "mean":
                vector1 = self._map_mean(sentence_1_preproc)
                vector2 = self._map_mean(sentence_2_preproc)
            elif mode == "embeddings":
                vector1 = self._map_word_embeddings(sentence_1, sequence_len, fixed_dictionary)
                vector2 = self._map_word_embeddings(sentence_2, sequence_len, fixed_dictionary)
            elif mode == "onehot":
                vector1 = self._map_one_hot(sentence_1_preproc)
                vector2 = self._map_one_hot(sentence_2_preproc)
            elif mode == "spacy":
                vector1 = self._map_spacy(sentence_1)
                vector2 = self._map_spacy(sentence_2)
            elif mode == "roberta":
                vector1 = self._map_roberta(sentence_1, self.rob)
                vector2 = self._map_roberta(sentence_2, self.rob)
            else:
                print("wrong mode")
            pares_vectores.append(((vector1, vector2), similitud))
        return pares_vectores
    
    def pair_list_to_x_y(self, pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        _x, _y = zip(*pair_list)
        _x_1, _x_2 = zip(*_x)
        return (np.array(_x_1), np.array(_x_2)), np.array(_y, dtype=np.float32, )
    

    def pretrained_weights(self):
        if self.pretrained:
            if self.remap_embeddings:
                self._pretrained_weights = np.zeros((len(self.diccionari.token2id) + 1, self.model.vector_size),  dtype=np.float32)
                for token, _id in self.diccionari.token2id.items():
                    if token in self.model:
                        self._pretrained_weights[_id + 1] = self.model[token]
                    else:
                        # In W2V, OOV will not have a representation. We will use 0.
                        pass
            else:
                # Not recommended (this will consume A LOT of RAM)
                self._pretrained_weights = np.zeros((self.model.vectors.shape[0] + 1, self.model.vector_size,),  dtype=np.float32)
                self._pretrained_weights[1:, :] = self.model.vectors
            

    def define_model(self):
        self.x_train, self.y_train = self.pair_list_to_x_y(self.mapped_train)
        self.x_val, self.y_val = self.pair_list_to_x_y(self.mapped_val)
        self.x_test, self.y_test = self.pair_list_to_x_y(self.mapped_test)
        batch_size: int = 64
        num_epochs: int = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((self.x_train, self.y_train))
        self.train_dataset = train_dataset.shuffle(buffer_size=len(self.x_train)).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((self.x_val, self.y_val))
        self.val_dataset = val_dataset.batch(batch_size)

        if self.mode != "embeddings":
            self.exec_model = self.build_and_compile_model( embedding_size= self.x_train[0].shape[1])
            #tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, )
            
        else:
            self.exec_model = self.build_and_compile_trainable_model(self.sequence_len, pretrained_weights=self._pretrained_weights, trainable=self.trainable)
        print(self.exec_model.summary())


    def build_and_compile_model(self, embedding_size: int = 300, learning_rate: float = 1e-3) -> tf.keras.Model:
        # Capa de entrada para los pares de vectores
        input_1 = tf.keras.Input(shape=(embedding_size,))
        input_2 = tf.keras.Input(shape=(embedding_size,))

        # Hidden layer
        first_projection = tf.keras.layers.Dense(
            embedding_size,
            kernel_initializer=tf.keras.initializers.Identity(),
            bias_initializer=tf.keras.initializers.Zeros(),
        )
        projected_1 = first_projection(input_1)
        projected_2 = first_projection(input_2)
        
        # Compute the cosine distance using a Lambda layer
        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))

        output = tf.keras.layers.Lambda(cosine_distance)([projected_1, projected_2])
        # Define output
        model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)

        # Compile the model
        model.compile(loss='mean_squared_error',
                    optimizer=tf.keras.optimizers.Adamax(learning_rate))
        return model
    
    def build_and_compile_trainable_model(self, input_length, dictionary_size = 1000, embedding_size = 300, pretrained_weights=None, learning_rate=1e-3, trainable=True):
        input_1 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)
        input_2 = tf.keras.Input(shape=(input_length,), dtype=tf.int32)

        if pretrained_weights is None:
            embedding = tf.keras.layers.Embedding(dictionary_size, embedding_size, input_length=input_length, mask_zero=True)
        else:
            dictionary_size = pretrained_weights.shape[0]
            embedding_size = pretrained_weights.shape[1]
            initializer = tf.keras.initializers.Constant(pretrained_weights)
            embedding = tf.keras.layers.Embedding(dictionary_size, embedding_size, input_length=input_length, mask_zero=True, embeddings_initializer=initializer, trainable=trainable)

        embedded_1 = embedding(input_1)
        embedded_2 = embedding(input_2)

        pooled_1 = tf.keras.layers.GlobalAveragePooling1D()(embedded_1)
        pooled_2 = tf.keras.layers.GlobalAveragePooling1D()(embedded_2)

        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))

        output = tf.keras.layers.Lambda(cosine_distance)([pooled_1, pooled_2])
        model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(learning_rate))
        return model
    # Define model 1


    def train(self, num_epochs=64):
        self.exec_model.fit(self.train_dataset, epochs=num_epochs, validation_data=self.val_dataset)
        print(f"Correlación de Pearson (baseline-train): {self.compute_pearson(self.x_train, self.y_train)}")
        print(f"Correlación de Pearson (baseline-validation): {self.compute_pearson(self.x_val, self.y_val)}")
        print(f"Correlación de Pearson (baseline-test): {self.compute_pearson(self.x_test, self.y_test)}")

    def compute_pearson_baseline(self, x_, y_):
        y_pred_baseline = []
        for v1, v2 in zip(*x_):
            d = 1.0 - spatial.distance.cosine(v1, v2)
            y_pred_baseline.append(d)
        # Calcular la correlación de Pearson entre las predicciones y los datos de prueba
        correlation, _ = pearsonr(y_pred_baseline, y_.flatten())
        return correlation
    

    def compute_pearson(self, x_, y_):
        # Obtener las predicciones del modelo para los datos de prueba. En este ejemplo vamos a utilizar el corpus de training.
        y_pred = self.exec_model.predict(x_)
        # Calcular la correlación de Pearson entre las predicciones y los datos de prueba
        correlation, _ = pearsonr(y_pred.flatten(), y_.flatten())
        return correlation
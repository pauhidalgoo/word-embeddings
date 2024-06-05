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
import pickle
from tensorflow_models import *
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import torch
from scipy.stats import spearmanr

import warnings

warnings.filterwarnings('ignore', category=FutureWarning)  # Suppress FutureWarnings
warnings.filterwarnings('ignore', category=UserWarning)    # Suppress UserWarnings
warnings.filterwarnings('ignore')


class TextSimilarity:
    """
    A class for text similarity computation using different models and embedding techniques.

    Attributes
    ----------
    input_pairs : List[Tuple[str, str, float]]
        List of sentence pairs and their similarity labels for training.
    input_pairs_val : List[Tuple[str, str, float]]
        List of sentence pairs and their similarity labels for validation.
    input_pairs_test : List[Tuple[str, str, float]]
        List of sentence pairs and their similarity labels for testing.
    model : Any
        Model used for obtaining embeddings.
    diccionari : Dictionary
        Gensim dictionary created from the input pairs.
    model_tfidf : TfidfModel
        TF-IDF model trained on the input pairs.
    _pretrained_weights : Optional[np.ndarray]
        Pretrained weights for embedding layer.

    Methods
    -------
    __simple_preprocess(sentence: str) -> List[str]:
        Preprocess a sentence into a list of tokens.
    create_dict_tfid(dict_size: int):
        Create a TF-IDF dictionary from the input pairs.
    _map_tf_idf(sentence_preproc: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        Map a preprocessed sentence to its TF-IDF weighted word embeddings.
    _map_word_embeddings(sentence: str, sequence_len: int = 96, fixed_dictionary: Optional[Dictionary] = None) -> np.ndarray:
        Map a sentence to word embedding indices.
    _map_mean(sentence: str) -> np.ndarray:
        Map a sentence to the mean of its word embeddings.
    _map_spacy(sentence: str) -> np.ndarray:
        Map a sentence using Spacy embeddings.
    _map_roberta_v2(sentence: str) -> np.ndarray:
        Map a sentence using RoBERTa model with the last hidden layer state.
    _map_roberta(sentence: str) -> np.ndarray:
        Map a sentence using RoBERTa model with the last hidden layer state.
    _map_roberta_hugging(sentence: str) -> np.ndarray:
        Map a sentence using Hugging Face RoBERTa model.
    _map_one_hot(sentence: List[str]) -> np.ndarray:
        Map a preprocessed sentence to one-hot encoded vector.
    map_pairs(sentence_pairs: List[Tuple[str, str, float]], dictionary: Dictionary = None, mode = None, tf_idf_model: TfidfModel = None, sequence_len: int = 96, fixed_dictionary: Optional[Dictionary] = None) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
        Map pairs of sentences to their corresponding vector representations.
    pair_list_to_x_y(pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        Convert a list of sentence pairs to input and output numpy arrays.
    pretrained_weights():
        Generate pretrained weights for embedding layer.
    define_model(id: int = 0):
        Define the model architecture based on the mode and selected model ID.
    train(num_epochs: int = 256):
        Train the model with the given number of epochs.
    baseline_model():
        Calculate Pearson and Spearman correlations for baseline model.
    compute_pearson_baseline(x_: Tuple[np.ndarray, np.ndarray], y_: np.ndarray) -> float:
        Compute Pearson correlation for baseline model.
    compute_pearson(x_: Tuple[np.ndarray, np.ndarray], y_: np.ndarray) -> float:
        Compute Pearson correlation for trained model.
    compute_spearman_baseline(x_: Tuple[np.ndarray, np.ndarray], y_: np.ndarray) -> float:
        Compute Spearman correlation for baseline model.
    compute_spearman(x_: Tuple[np.ndarray, np.ndarray], y_: np.ndarray) -> float:
        Compute Spearman correlation for trained model.
    save_mapped_pairs(mode: str):
        Save mapped sentence pairs to a pickle file.
    load_mapped_pairs(mode: str, recalculate: bool):
        Load mapped sentence pairs from a pickle file.
    """

    def __init__(self, model, dataset, remap_embeddings=None, mode="mean", cls=True, pretrained=False, remap=True, trainable=True, dict_size=15000, tokenizer=None, recalculate=True):
        """
        Initialize the TextSimilarity object.

        Parameters
        ----------
        model : Any
            The model used for obtaining embeddings.
        dataset : Any
            The dataset containing the sentence pairs.
        remap_embeddings : Optional[str]
            Whether to remap embeddings.
        mode : str, optional
            The mode of embedding mapping ('mean', 'tfidf', 'embeddings', etc.).
        cls : bool, optional
            Whether to use CLS token.
        pretrained : bool, optional
            Whether to use pretrained weights.
        remap : bool, optional
            Whether to remap embeddings.
        trainable : bool, optional
            Whether embeddings are trainable.
        dict_size : int, optional
            Size of the dictionary.
        tokenizer : Optional[Callable]
            Tokenizer function.
        recalculate : bool, optional
            Whether to recalculate embeddings.
        """
        self.input_pairs = [(e["sentence1"], e["sentence2"], e["label"]) for e in dataset["train"].to_list()]
        self.input_pairs_val = [(e["sentence1"], e["sentence2"], e["label"]) for e in dataset["validation"].to_list()]
        self.input_pairs_test = [(e["sentence1"], e["sentence2"], e["label"]) for e in dataset["test"].to_list()]
        self.model = model
        self.create_dict_tfid(dict_size=dict_size)
        self.cls = cls
        self.pretrained = pretrained
        self.remap_embeddings = remap
        self.mode = mode
        self.trainable = trainable
        self.tokenizer = tokenizer
        if recalculate == True:
            self.mapped_train = self.map_pairs(self.input_pairs, dictionary=self.diccionari, mode=mode)
            self.mapped_val = self.map_pairs(self.input_pairs_val, dictionary=self.diccionari, mode=mode)
            self.mapped_test = self.map_pairs(self.input_pairs_test, dictionary=self.diccionari, mode=mode)
            self.save_mapped_pairs(mode)
        else:
            self.load_mapped_pairs(mode, recalculate)
        self._pretrained_weights: Optional[np.ndarray] = None
        if self.pretrained and mode == "embeddings":
            self.pretrained_weights()

    def __simple_preprocess(self, sentence: str) -> List[str]:
        """
        Preprocess a sentence into a list of tokens.

        Parameters
        ----------
        sentence : str
            The sentence to preprocess.

        Returns
        -------
        List[str]
            The preprocessed sentence as a list of tokens.
        """
        preprocessed = simple_preprocess(sentence)
        return preprocessed

    def create_dict_tfid(self, dict_size: int):
        """
        Create a TF-IDF dictionary from the input pairs.

        Parameters
        ----------
        dict_size : int
            The size of the dictionary.
        """
        all_input_pairs = self.input_pairs + self.input_pairs_val + self.input_pairs_test
        sentences_1_preproc = [simple_preprocess(sentence_1) for sentence_1, _, _ in all_input_pairs]
        sentences_2_preproc = [simple_preprocess(sentence_2) for _, sentence_2, _ in all_input_pairs]
        sentence_pairs = list(zip(sentences_1_preproc, sentences_2_preproc))
        sentences_pairs_flattened = sentences_1_preproc + sentences_2_preproc
        self.diccionari = Dictionary(sentences_pairs_flattened)
        corpus = [self.diccionari.doc2bow(sent) for sent in sentences_pairs_flattened]
        self.model_tfidf = TfidfModel(corpus)

    def _map_tf_idf(self, sentence_preproc: List[str]) -> Tuple[List[np.ndarray], List[float]]:
        """
        Map a preprocessed sentence to its TF-IDF weighted word embeddings.

        Parameters
        ----------
        sentence_preproc : List[str]
            The preprocessed sentence.

        Returns
        -------
        Tuple[List[np.ndarray], List[float]]
            Vectors and weights of the sentence.
        """
        bow = self.diccionari.doc2bow(sentence_preproc)
        tf_idf = self.model_tfidf[bow]
        vectors, weights = [], []
        for word_index, weight in tf_idf:
            word = self.diccionari.get(word_index)
            if word in self.model:
                vectors.append(self.model[word])
                weights.append(weight)
        return np.average(vectors, weights=weights, axis=0)

    def _map_word_embeddings(self, sentence: str, sequence_len: int = 96, fixed_dictionary: Optional[Dictionary] = None) -> np.ndarray:
        """
        Map a sentence to word embedding
         Parameters
        ----------
        sentence : str
            The input sentence.
        sequence_len : int, optional
            The maximum sequence length, by default 96.
        fixed_dictionary : Optional[Dictionary], optional
            A fixed dictionary, by default None.

        Returns
        -------
        np.ndarray
            Word embedding indices.
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
    
    def _map_mean(self, sentence: str) -> np.ndarray:
        """
        Map a sentence to the mean of its word embeddings.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        np.ndarray
            The mean of word embeddings.
        """
        vector = [self.model[word] for word in sentence if word in self.model]
        return np.mean(vector, axis=0)
    
    def _map_spacy(self, sentence: str) -> np.ndarray:
        """
        Map a sentence using SpaCy embeddings.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        np.ndarray
            The mapped vector using SpaCy embeddings.
        """
        sent = self.model(sentence)
        return sent.vector
    
    def _map_roberta_v2(self, sentence: str) -> np.ndarray:
        """
        Map a sentence using RoBERTa model with the last hidden layer state (interprets CLS position 0).

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        np.ndarray
            The mapped vector using RoBERTa model.
        """
        doc = self.model(sentence)
        if self.cls:
            return doc._.trf_data.last_hidden_layer_state.data[0]
        else:
            return np.mean(doc._.trf_data.last_hidden_layer_state.data[1:], axis=0)
        
    def _map_roberta(self, sentence: str) -> np.ndarray:
        """
        Map a sentence using RoBERTa model with the last hidden layer state (interprets CLS position -1).

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        np.ndarray
            The mapped vector using RoBERTa model.
        """
        doc = self.model(sentence)
        if self.cls:
            return doc._.trf_data.last_hidden_layer_state.data[-1]
        else:
            return np.mean(doc._.trf_data.last_hidden_layer_state.data[:-1], axis=0)
        
    def _map_roberta_hugging(self, sentence: str) -> np.ndarray:
        """
        Map a sentence using Hugging Face RoBERTa model.

        Parameters
        ----------
        sentence : str
            The input sentence.

        Returns
        -------
        np.ndarray
            The mapped vector using RoBERTa model.
        """
        sentence = self.tokenizer(sentence, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**sentence)
        if self.cls:
            return outputs.last_hidden_state[:, 0, :].squeeze().numpy()
        else:
            return  outputs.last_hidden_state.mean(dim=1).squeeze().numpy()

        
    def _map_one_hot(self, sentence: List[str]) -> np.ndarray:
        """
        Map a preprocessed sentence to one-hot encoded vector.

        Parameters
        ----------
        sentence : List[str]
            The preprocessed sentence.

        Returns
        -------
        np.ndarray
            The one-hot encoded vector.
        """
        new_dict = self.diccionari
        new_dict.filter_extremes(no_below=5, no_above=1, keep_n=1000)
        vector = np.zeros(len(new_dict), dtype = np.float32)
        for word_index, _ in new_dict.doc2bow(sentence):
            vector[word_index] = 1
        return vector

    def map_pairs(self, sentence_pairs: List[Tuple[str, str, float]], dictionary: Dictionary = None, mode = None, tf_idf_model: TfidfModel = None, sequence_len: int = 96, fixed_dictionary: Optional[Dictionary] = None) -> List[Tuple[Tuple[np.ndarray, np.ndarray], float]]:
        """
        Map pairs of sentences to their corresponding vector representations.

        Parameters
        ----------
        sentence_pairs : List[Tuple[str, str, float]]
            List of sentence pairs and their similarity labels.
        dictionary : Dictionary, optional
            Gensim dictionary, by default None.
        mode : str, optional
            Embedding mapping mode, by default None.
        tf_idf_model : TfidfModel, optional
            TF-IDF model, by default None.
        sequence_len : int, optional
            Maximum sequence length, by default 96.
        fixed_dictionary : Optional[Dictionary], optional
            Fixed dictionary, by default None.

        Returns
        -------
        List[Tuple[Tuple[np.ndarray, np.ndarray], float]]
            List of tuples containing sentence vectors and their similarity labels.
        """
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
                vector1 = self._map_word_embeddings(sentence_1, sequence_len, self.diccionari)
                vector2 = self._map_word_embeddings(sentence_2, sequence_len,  self.diccionari)
            elif mode == "onehot":
                vector1 = self._map_one_hot(sentence_1_preproc)
                vector2 = self._map_one_hot(sentence_2_preproc)
            elif mode == "spacy":
                vector1 = self._map_spacy(sentence_1)
                vector2 = self._map_spacy(sentence_2)
            elif mode == "roberta":
                vector1 = self._map_roberta(sentence_1)
                vector2 = self._map_roberta(sentence_2)
            elif mode == "roberta2":
                vector1 = self._map_roberta_v2(sentence_1)
                vector2 = self._map_roberta_v2(sentence_2)
            elif mode == "roberta-hugging":
                vector1 = self._map_roberta_hugging(sentence_1)
                vector2 = self._map_roberta_hugging(sentence_2)
            else:
                print("wrong mode")
            pares_vectores.append(((vector1, vector2), similitud))
        return pares_vectores
    
    def pair_list_to_x_y(self, pair_list: List[Tuple[Tuple[np.ndarray, np.ndarray], int]]) -> Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]:
        """
        Convert a list of sentence pairs and their similarity labels into input-output pairs for model training.

        Parameters
        ----------
        pair_list : List[Tuple[Tuple[np.ndarray, np.ndarray], int]]
            List of sentence pairs with their similarity labels.

        Returns
        -------
        Tuple[Tuple[np.ndarray, np.ndarray], np.ndarray]
            Tuple containing input pairs and their labels.
        """
        _x, _y = zip(*pair_list)
        _x_1, _x_2 = zip(*_x)
        return (np.array(_x_1), np.array(_x_2)), np.array(_y, dtype=np.float32, )
    

    def pretrained_weights(self):
        """
        Compute pretrained weights if specified and update the instance attribute.
        """
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
            

    def define_model(self, id:int = 0):
        """
        Define the model for training based on the selected architecture and mode.
        Sets self.exec_model
        Id should be between 0 and 7.
        """
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
            if id == 0:
                self.exec_model = model_1()
            elif id == 1:
                self.exec_model = model_2(embedding_size= self.x_train[0].shape[1])
            elif id == 2:
                self.exec_model = model_3(embedding_size= self.x_train[0].shape[1])
            elif id == 3:
                self.exec_model = model_4()
            elif id == 4:
                self.exec_model = model_5(embedding_size= self.x_train[0].shape[1])
            elif id == 5:
                self.exec_model = model_6(embedding_size= self.x_train[0].shape[1])
            elif id == 6:
                self.exec_model = model_7()
            else:
                self.exec_model = model_8(embedding_size= self.x_train[0].shape[1])
            #tf.keras.utils.plot_model(model, show_shapes=True, show_layer_activations=True, )
            
        else:
            if id == 0:
                self.exec_model = model_embeddings_1(self.sequence_len, dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=False)
            elif id == 1:
                self.exec_model = model_embeddings_1(self.sequence_len, dictionary_size= len(self.diccionari) +1,pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=True)
            elif id == 2:
                self.exec_model = model_embeddings_2(self.sequence_len,dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=False)
            elif id == 3:
                self.exec_model = model_embeddings_2(self.sequence_len,dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=True)
            elif id == 4:
                self.exec_model = model_embeddings_3(self.sequence_len, dictionary_size= len(self.diccionari) +1,pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=False)
            elif id == 5:
                self.exec_model = model_embeddings_3(self.sequence_len,dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable, use_cosine=True)
            elif id == 6:
                self.exec_model = model_embeddings_7(self.sequence_len,dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable)
            else:
                self.exec_model = model_embeddings_8(self.sequence_len,dictionary_size= len(self.diccionari) +1, pretrained_weights=self._pretrained_weights, trainable=self.trainable)

        #print(self.exec_model.summary())


    def train(self, num_epochs:int=256) -> Tuple[float, float, float, float, float, float]:
        """
        Train the defined model.

        Parameters
        ----------
        num_epochs : int, optional
            Number of epochs for training, by default 256.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Training and validation metrics.
        """
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            patience=25,        
            verbose=0,      
            restore_best_weights=True  
        )

        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',  
            factor=0.1,         
            patience=10,         
            verbose=0,        
            min_lr=1e-7     
        )
        self.exec_model.fit(self.train_dataset, epochs=num_epochs, validation_data=self.val_dataset, callbacks=[early_stopping, reduce_lr],  verbose=0)
        train_pearson = self.compute_pearson(self.x_train, self.y_train)
        val_pearson = self.compute_pearson(self.x_val, self.y_val)
        test_pearson = self.compute_pearson(self.x_test, self.y_test)
        #print(f"Correlación de Pearson (train): {train_pearson}")
        #print(f"Correlación de Pearson (validation): {val_pearson}")
        #print(f"Correlación de Pearson (test): {test_pearson}")

        train_spearman = self.compute_spearman(self.x_train, self.y_train)
        val_spearman = self.compute_spearman(self.x_val, self.y_val)
        test_spearman = self.compute_spearman(self.x_test, self.y_test)

        return train_pearson, val_pearson, test_pearson, train_spearman, val_spearman, test_spearman


    def baseline_model(self) -> Tuple[float, float, float, float, float, float]:
        """
        Train and evaluate the baseline model.

        Returns
        -------
        Tuple[float, float, float, float, float, float]
            Training and validation metrics of the baseline model.
        """
        train_pearson = self.compute_pearson_baseline(self.x_train, self.y_train)
        val_pearson = self.compute_pearson_baseline(self.x_val, self.y_val)
        test_pearson = self.compute_pearson_baseline(self.x_test, self.y_test)
        #print(f"Correlación de Pearson (baseline-train): {train_pearson}")
        #print(f"Correlación de Pearson (baseline-validation): {val_pearson}")
        #print(f"Correlación de Pearson (baseline-test): {test_pearson}")

        train_spearman= self.compute_spearman_baseline(self.x_train, self.y_train)
        val_spearman = self.compute_spearman_baseline(self.x_val, self.y_val)
        test_spearman = self.compute_spearman_baseline(self.x_test, self.y_test)

        return train_pearson, val_pearson, test_pearson, train_spearman, val_spearman, test_spearman

    def compute_pearson_baseline(self, x_:Tuple[np.ndarray, np.ndarray], y_:np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient for the baseline model.

        Parameters
        ----------
        x_ : Tuple[np.ndarray, np.ndarray]
            Input pairs.
        y_ : np.ndarray
            Labels.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        y_pred_baseline = []
        for v1, v2 in zip(*x_):
            d = 1.0 - spatial.distance.cosine(v1, v2)
            y_pred_baseline.append(d)
        # Calcular la correlación de Pearson entre las predicciones y los datos de prueba
        correlation, _ = pearsonr(y_pred_baseline, y_.flatten())
        return correlation
    

    def compute_pearson(self, x_:Tuple[np.ndarray, np.ndarray], y_:np.ndarray) -> float:
        """
        Compute Pearson correlation coefficient for the trained model.

        Parameters
        ----------
        x_ : Tuple[np.ndarray, np.ndarray]
            Input pairs.
        y_ : np.ndarray
            Labels.

        Returns
        -------
        float
            Pearson correlation coefficient.
        """
        # Obtener las predicciones del modelo para los datos de prueba. En este ejemplo vamos a utilizar el corpus de training.
        y_pred = self.exec_model.predict(x_, verbose=0)
        # Calcular la correlación de Pearson entre las predicciones y los datos de prueba
        correlation, _ = pearsonr(y_pred.flatten(), y_.flatten())
        return correlation
    
    def compute_spearman_baseline(self, x_:Tuple[np.ndarray, np.ndarray], y_:np.ndarray) -> float:
        """
        Compute Spearman correlation coefficient for the baseline model.

        Parameters
        ----------
        x_ : Tuple[np.ndarray, np.ndarray]
            Input pairs.
        y_ : np.ndarray
            Labels.

        Returns
        -------
        float
            Spearman correlation coefficient.
        """
        y_pred_baseline = []
        for v1, v2 in zip(*x_):
            d = 1.0 - spatial.distance.cosine(v1, v2)
            y_pred_baseline.append(d)
        correlation, _ = spearmanr(y_pred_baseline, y_.flatten())
        return correlation
    
    def compute_spearman(self, x_:Tuple[np.ndarray, np.ndarray], y_:np.ndarray) -> float:
        """
        Compute Spearman correlation coefficient for the trained model.

        Parameters
        ----------
        x_ : Tuple[np.ndarray, np.ndarray]
            Input pairs.
        y_ : np.ndarray
            Labels.

        Returns
        -------
        float
            Spearman correlation coefficient.
        """
        y_pred = self.exec_model.predict(x_, verbose=0)
        correlation, _ = spearmanr(y_pred.flatten(), y_.flatten())
        return correlation

    def save_mapped_pairs(self, mode:str):
        """
        Save mapped pairs to a file.

        Parameters
        ----------
        mode : str
            Mode of mapping.
        """
        with open(f'./data/{mode}_mapped_pairs.pkl', 'wb') as f:
            pickle.dump((self.mapped_train, self.mapped_val, self.mapped_test), f)

    def load_mapped_pairs(self, mode, recalculate:str):
        """
        Load mapped pairs from a file.

        Parameters
        ----------
        mode : str
            Mode of mapping.
        recalculate : bool
            Whether to recalculate or not.
        """
        with open(f'./data/{recalculate}.pkl', 'rb') as f:
            self.mapped_train, self.mapped_val, self.mapped_test = pickle.load(f)
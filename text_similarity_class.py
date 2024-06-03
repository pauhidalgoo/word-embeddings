import numpy as np
from scipy import spatial
from typing import Tuple, List, Optional
from datasets import load_dataset
import tensorflow as tf
from gensim.models import TfidfModel, fasttext
from gensim.utils import simple_preprocess
from gensim.corpora import Dictionary
from scipy.stats import pearsonr
import spacy

def preprocess(sentence: str) -> List[str]:
    return simple_preprocess(sentence)


class TextSimilarity:
    def __init__(self, word_embedding_model, spacy_model=None):
        self.wv_model = word_embedding_model
        self.spacy_model = spacy_model
        self.dictionary = None
        self.tfidf_model = None
        self.max_len = 100

    def prepare_data(self, input_pairs):
        sentences_1_preproc = [simple_preprocess(sentence_1) for sentence_1, _, _ in input_pairs]
        sentences_2_preproc = [simple_preprocess(sentence_2) for _, sentence_2, _ in input_pairs]
        sentences_pairs_flattened = sentences_1_preproc + sentences_2_preproc
        self.dictionary = Dictionary(sentences_pairs_flattened)
        corpus = [self.dictionary.doc2bow(sent) for sent in sentences_pairs_flattened]
        self.tfidf_model = TfidfModel(corpus)
    
    def map_tf_idf(self, sentence_preproc):
        bow = self.dictionary.doc2bow(sentence_preproc)
        tf_idf = self.tfidf_model[bow]
        vectors, weights = [], []
        for word_index, weight in tf_idf:
            word = self.dictionary.get(word_index)
            if word in self.wv_model:
                vectors.append(self.wv_model[word])
                weights.append(weight)
        return vectors, weights

    def map_pairs(self, sentence_pairs, use_tfidf=False, use_roberta=False):
        mapped_pairs = []
        for sentence_1, sentence_2, similitud in sentence_pairs:
            if use_roberta:
                vector1 = self.get_roberta_embeddings(sentence_1)
                vector2 = self.get_roberta_embeddings(sentence_2)
            else:
                sentence_1_preproc = preprocess(sentence_1)
                sentence_2_preproc = preprocess(sentence_2)
                if use_tfidf:
                    vectors1, weights1 = self.map_tf_idf(sentence_1_preproc)
                    vectors2, weights2 = self.map_tf_idf(sentence_2_preproc)
                    vector1 = np.average(vectors1, weights=weights1, axis=0)
                    vector2 = np.average(vectors2, weights=weights2, axis=0)
                else:
                    vectors1 = [self.wv_model[word] for word in sentence_1_preproc if word in self.wv_model]
                    vectors2 = [self.wv_model[word] for word in sentence_2_preproc if word in self.wv_model]
                    vector1 = np.mean(vectors1, axis=0)
                    vector2 = np.mean(vectors2, axis=0)
            mapped_pairs.append(((vector1, vector2), similitud))
        return mapped_pairs

    def get_roberta_embeddings(self, sentence: str) -> np.ndarray:
        doc = self.spacy_model(sentence)
        # Use the CLS token or mean pooling of token embeddings
        return doc._.trf_data.last_hidden_layer_state[-1].data.mean(axis=0)

    def pair_list_to_x_y(self, pair_list):
        _x, _y = zip(*pair_list)
        _x_1, _x_2 = zip(*_x)
        return (np.array(_x_1), np.array(_x_2)), np.array(_y, dtype=np.float32)

    def prepare_datasets(self, mapped_train, mapped_val, mapped_test, batch_size):
        x_train, y_train = self.pair_list_to_x_y(mapped_train)
        x_val, y_val = self.pair_list_to_x_y(mapped_val)
        x_test, y_test = self.pair_list_to_x_y(mapped_test)

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(len(x_train)).batch(batch_size)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val)).batch(batch_size)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

        return train_dataset, val_dataset, test_dataset

    def build_and_compile_model(self, embedding_size=100, learning_rate=1e-3):
        input_1 = tf.keras.Input(shape=(embedding_size,))
        input_2 = tf.keras.Input(shape=(embedding_size,))

        first_projection = tf.keras.layers.Dense(embedding_size, kernel_initializer=tf.keras.initializers.Identity(), bias_initializer=tf.keras.initializers.Zeros())
        projected_1 = first_projection(input_1)
        projected_2 = first_projection(input_2)

        def cosine_distance(x):
            x1, x2 = x
            x1_normalized = tf.keras.backend.l2_normalize(x1, axis=1)
            x2_normalized = tf.keras.backend.l2_normalize(x2, axis=1)
            return 2.5 * (1.0 + tf.reduce_sum(x1_normalized * x2_normalized, axis=1))

        output = tf.keras.layers.Lambda(cosine_distance)([projected_1, projected_2])
        model = tf.keras.Model(inputs=[input_1, input_2], outputs=output)
        model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adamax(learning_rate))
        return model
    
    

    def train_model(self, model, train_dataset, val_dataset, num_epochs):
        model.fit(train_dataset, epochs=num_epochs, validation_data=val_dataset)

    def compute_pearson(self, model, x_, y_):
        y_pred = model.predict(x_)
        correlation, _ = pearsonr(y_pred.flatten(), y_.flatten())
        return correlation

    def build_and_compile_trainable_model(self, input_length, dictionary_size, embedding_size, pretrained_weights=None, learning_rate=1e-3, trainable=True):
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

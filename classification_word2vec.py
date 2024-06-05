import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import os

from datasets import load_dataset
from gensim.models import Word2Vec
import spacy
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adamax, Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import warnings
from typing import Callable

class ClassificationWord2Vec:
	def __init__(self, 
		predict_labels: int=1, 
		w2v_model_name: str="w2v_sg_300_win10_nltk_cat_gc_1000mb.model", 
		tokenizer: str|None=None,
		) -> None:
		"""
		Constructor of the class.

		Parameters
		----------
		predict_labels: int
			If the labels to classify are label1 or label2.
			
			label1 contains 4 classes: 'Societat', 'Política', 'Economia', 'Cultura'.
			
			label2 contains 53 classes:
			'Llengua', 'Infraestructures', 'Arts', 'Parlament', 'Noves tecnologies', 'Castells', 
			'Successos', 'Empresa', 'Mobilitat', 'Teatre', 'Treball', 'Logística', 'Urbanisme', 
			'Govern', 'Entitats', 'Finances', 'Govern espanyol', 'Trànsit', 'Indústria', 'Esports', 
			'Exteriors', 'Medi ambient', 'Habitatge', 'Salut', 'Equipaments i patrimoni', 'Recerca', 
			'Cooperació', 'Innovació', 'Agroalimentació', 'Policial', 'Serveis Socials', 'Cinema', 
			'Memòria històrica', 'Turisme', 'Política municipal', 'Comerç', 'Universitats', 'Hisenda', 
			'Judicial', 'Partits', 'Música', 'Lletres', 'Religió', 'Festa i cultura popular', 
			'Unió Europea', 'Moda', 'Moviments socials', 'Comptes públics', 'Immigració', 
			'Educació', 'Gastronomia', 'Meteorologia', 'Energia'
		w2v_model_name: str
			Name of the embeddings Word2Vec model in the './models/word2vec/' folder.
		tokenizer: str
			Tokenizer to use. If None, the tokenizer is inferred from the embeddings model name.
		"""
		assert predict_labels in [1, 2], "predict_labels must be 1 or 2."
		assert os.path.exists(f"./models/word2vec/{w2v_model_name}"), f"Word2Vec model '{w2v_model_name}' not found in './models/word2vec/' folder."

		self.embeddings_model = self.__load_embeddings_model(w2v_model_name=w2v_model_name)
		self.vector_size = self.embeddings_model.vector_size
		self.tokenizer = self.__set_tokenizer(model_name=w2v_model_name, tokenizer_name=tokenizer)
		self.X_train, self.X_val, self.X_test, self.y_train, self.y_val, self.y_test = self.__load_raw_data(predict_labels=predict_labels)

		self.num_classes = len(np.unique(np.argmax(self.y_train, axis=1)))
		print(f"Number of classes: {self.num_classes}")
		print(f"Percentage of classes in the training set: {np.unique(np.argmax(self.y_train, axis=1), return_counts=True)[1] / len(self.y_train)}")

		self.model = None
		self.trained = False

	def __load_embeddings_model(self, w2v_model_name: str) -> Word2Vec:
		"""
		Load embeddings model.

		Parameters
		----------
		w2v_model_name: str
			Name of the embeddings Word2Vec model in the './models/word2vec/' folder.

		Returns
		-------
		Word2Vec
			Word2Vec model.
		"""
		assert os.path.exists(f"./models/word2vec/{w2v_model_name}"), f"Word2Vec model '{w2v_model_name}' not found in './models/word2vec/' folder."
		
		print("Loading embeddings model...")

		return Word2Vec.load(f"./models/word2vec/{w2v_model_name}")

	def __set_tokenizer(self, model_name: str, tokenizer_name: str|None=None) -> Callable:
		"""
		Set tokenizer for the sentences.

		Parameters
		----------
		model_name: str
			Name of the embeddings model.
		tokenizer_name: str
			Tokenizer to use. If None, the tokenizer is inferred from the embeddings model name.

		Returns
		-------
		Callable
			Tokenizer function.
		"""
		assert tokenizer_name in [None, "nltk", "spacy"], "Tokenizer not valid. Valid tokenizers: None, 'nltk', 'spacy'."

		print("Setting tokenizer...")
		
		if tokenizer_name is None: # Infer tokenizer from the embeddings model name
			assert "nltk" in model_name or "spacy" in model_name, "Tokenizer not inferred from the embeddings model name. Please, set the tokenizer manually in the constructor."
			if "nltk" in model_name:
				tokenizer_name = "nltk"
			elif "spacy" in model_name:
				tokenizer_name = "spacy"

		if tokenizer_name == "nltk":
			tokenizer = nltk.word_tokenize

		elif tokenizer_name == "spacy":
			self.nlp = spacy.load("ca_core_news_sm")
			tokenizer = lambda sentence: [token.text for token in self.nlp(sentence)]

		self.tokenizer_name = tokenizer_name

		return tokenizer

	def __load_raw_data(self, predict_labels: int) -> tuple:
		"""
		Load dataset.

		Parameters
		----------
		predict_labels: int
			If the labels to classify are labels_1 or labels_2.		

		Returns
		-------
		tuple
			X_train, X_val, X_test, y_train, y_val, y_test
		"""
		assert predict_labels in [1, 2], "predict_labels must be 1 or 2."

		print("Loading dataset...")

		bool_X_train = False
		bool_X_val = False
		bool_X_test = False
		bool_y_train = False
		bool_y_val = False
		bool_y_test = False

		X_train, X_val, X_test, y_train, y_val, y_test = None, None, None, None, None, None

		if os.path.exists(f"./data/x_train_tecla_{self.tokenizer_name}.pkl"):
			X_train = pkl.load(open(f"./data/x_train_tecla_{self.tokenizer_name}.pkl", "rb"))
			bool_X_train = True

		if os.path.exists(f"./data/x_val_tecla_{self.tokenizer_name}.pkl"):
			X_val = pkl.load(open(f"./data/x_val_tecla_{self.tokenizer_name}.pkl", "rb"))
			bool_X_val = True

		if os.path.exists(f"./data/x_test_tecla_{self.tokenizer_name}.pkl"):
			X_test = pkl.load(open(f"./data/x_test_tecla_{self.tokenizer_name}.pkl", "rb"))
			bool_X_test = True

		if os.path.exists(f"./data/y_train_tecla_{self.tokenizer_name}_label{predict_labels}.pkl"):
			y_train = pkl.load(open(f"./data/y_train_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "rb"))
			bool_y_train = True

		if os.path.exists(f"./data/y_val_tecla_{self.tokenizer_name}_label{predict_labels}.pkl"):
			y_val = pkl.load(open(f"./data/y_val_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "rb"))
			bool_y_val = True

		if os.path.exists(f"./data/y_test_tecla_{self.tokenizer_name}_label{predict_labels}.pkl"):
			y_test = pkl.load(open(f"./data/y_test_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "rb"))
			bool_y_test = True

		if all([bool_X_train, bool_X_val, bool_X_test, bool_y_train, bool_y_val, bool_y_test]):
			return X_train, X_val, X_test,  y_train, y_val, y_test

		# Load dataset
		dataset = load_dataset("projecte-aina/tecla", trust_remote_code=True)

		train = dataset["train"]
		val = dataset["validation"]
		test = dataset["test"]

		if not bool_X_train:
			X_train = train["text"]

		if not bool_X_val:
			X_val = val["text"]

		if not bool_X_test:
			X_test = test["text"]

		if not bool_y_train:
			if predict_labels == 1:
				y_train = train["label1"]
			elif predict_labels == 2:
				y_train = train["label2"]

			self.num_classes = len(np.unique(y_train))
		else:
			self.num_classes = len(np.unique(np.argmax(y_train, axis=1)))


		if not bool_y_val:
			if predict_labels == 1:
				y_val = val["label1"]
			elif predict_labels == 2:
				y_val = val["label2"]

		if not bool_y_test:
			if predict_labels == 1:
				y_test = test["label1"]
			elif predict_labels == 2:
				y_test = test["label2"]

		# Preprocess the sentences and save them if they are not already saved
		if not bool_X_train:
			print("Preprocessing train sentences...")
			X_train = self.__preprocess_sentences(X_train)
			X_train = np.array(X_train)
			pkl.dump(X_train, open(f"./data/x_train_tecla_{self.tokenizer_name}.pkl", "wb"))

		if not bool_X_val:
			print("Preprocessing val sentences...")
			X_val = self.__preprocess_sentences(X_val)
			X_val = np.array(X_val)
			pkl.dump(X_val, open(f"./data/x_val_tecla_{self.tokenizer_name}.pkl", "wb"))

		if not bool_X_test:
			print("Preprocessing test sentences...")
			X_test = self.__preprocess_sentences(X_test)
			X_test = np.array(X_test)
			pkl.dump(X_test, open(f"./data/x_test_tecla_{self.tokenizer_name}.pkl", "wb"))

		if not bool_y_train:
			y_train = np.array(y_train)
			y_train = to_categorical(y_train, num_classes=self.num_classes)
			pkl.dump(y_train, open(f"./data/y_train_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "wb"))

		if not bool_y_val:
			y_val = np.array(y_val)
			y_val = to_categorical(y_val, num_classes=self.num_classes)
			pkl.dump(y_val, open(f"./data/y_val_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "wb"))

		if not bool_y_test:
			y_test = np.array(y_test)
			y_test = to_categorical(y_test, num_classes=self.num_classes)
			pkl.dump(y_test, open(f"./data/y_test_tecla_{self.tokenizer_name}_label{predict_labels}.pkl", "wb"))

		return X_train, X_val, X_test, y_train, y_val, y_test
	
	def __preprocess_sentences(self, sentences: list[str]):
		"""
		Preprocess the sentences.

		Parameters
		----------
		sentences: list[str]
			Sentences to preprocess.

		Returns
		-------
		list[str]
			Preprocessed sentences.
		"""
		sentences = [self.tokenizer(sentence) for sentence in sentences]
		sentences = [[token.lower() for token in sentence] for sentence in sentences]
		sentences = [self.__map_sentence(sentence) for sentence in sentences]

		return sentences
	
	def __map_sentence(self, sentence: list[str]) -> np.ndarray:
		"""
		Map a sentence to a vector of the mean of the embeddings of the words.

		Parameters
		----------
		sentence: str
			Sentence to map. Must be preprocessed and tokenized.
		"""
		mapped_sentence = []
		for word in sentence:
			try:
				mapped_sentence.append(self.embeddings_model.wv[word])
			except KeyError:
				warnings.warn(f"Word '{word}' not found in the embeddings model. It will be ignored.")
				continue

		return np.mean(mapped_sentence, axis=0)
	
	def build_model(self, 
		hidden_layers: list[int]=[512, 256, 128, 64],
		activation: str="relu",
		optimizer: str="adam",
		learning_rate: float=0.001
		) -> None:
		"""
		Build the model.

		Parameters
		----------
		hidden_layers: list[int]
			Number of neurons in each hidden layer.
		activation: str
			Activation function for the hidden layers.
		optimizer: str
			Optimizer to use. Options: 'adam', 'adamax'.
		learning_rate: float
			Learning rate of the optimizer.
		"""
		assert optimizer in ["adam", "adamax"], "Optimizer not valid. Valid optimizers: 'adam', 'adamax'."

		model = Sequential(name="ClassificationWord2Vec")
		
		model.add(Input(shape=(self.vector_size,)))
		
		for neurons in hidden_layers:
			model.add(Dense(neurons, activation=activation))

		model.add(Dense(self.num_classes, activation='softmax'))

		if optimizer == "adam":
			optimizer = Adam(learning_rate=learning_rate)
		elif optimizer == "adamax":
			optimizer = Adamax(learning_rate=learning_rate)

		model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

		print(model.summary())

		self.model = model

	def train_model(self, class_weights: bool=False) -> None:
		"""
		Train the model and plot the training and validation loss and accuracy.

		Parameters
		----------
		class_weights: bool
			If the class weights are used in the loss function to balance the classes in the training set.
		"""
		assert self.model is not None, "Model not defined. Please, build the model before training it."

		print(f"Shape of X_train: {self.X_train.shape}")
		print(f"Shape of y_train: {self.y_train.shape}")
		print(f"Shape of X_val: {self.X_val.shape}")
		print(f"Shape of y_val: {self.y_val.shape}")

		train_class_weights = {i: 1 for i in range(self.num_classes)}
		if class_weights:
			y_integers = np.argmax(self.y_train, axis=1)
			class_weights_dict = {i: 1.0 / np.sum(y_integers == i) for i in range(self.num_classes)}
			train_class_weights = {i: class_weights_dict[i] / min(class_weights_dict.values()) for i in class_weights_dict}
			
			print(f"Class weights: {train_class_weights}")

		early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1)

		self.model.fit(self.X_train, self.y_train, validation_data=(self.X_val, self.y_val), epochs=500, batch_size=256, callbacks=[early_stopping], class_weight=train_class_weights)
		self.trained = True

		self.__plot_training_curves(self.model.history)

	def __plot_training_curves(self, history) -> None:
		"""
		Plot the training and validation loss and accuracy.

		Parameters
		----------
		history: History
			History object returned by the fit method.
		"""
		# Plot the training and validation loss and accuracy
		fig, ax = plt.subplots(2, 1, figsize=(8, 8))
		ax[0].plot(history.history['loss'], label='train_loss')
		ax[0].plot(history.history['val_loss'], label='val_loss')
		ax[0].set_xlabel('Epoch')
		ax[0].set_ylabel('Loss')
		ax[0].set_title('Training and Validation Loss')
		ax[0].grid(True)
		ax[0].legend()
		ax[1].plot(history.history['accuracy'], label='train_acc')
		ax[1].plot(history.history['val_accuracy'], label='val_acc')
		ax[1].set_xlabel('Epoch')
		ax[1].set_ylabel('Accuracy')
		ax[1].set_title('Training and Validation Accuracy')
		ax[1].grid(True)
		ax[1].legend()
		plt.tight_layout()
		plt.show()

		# Print the values of the epoch with the best validation loss
		best_epoch = np.argmin(history.history['val_loss'])
		print(f"Train loss: {history.history['loss'][best_epoch]}")
		print(f"Validation loss: {history.history['val_loss'][best_epoch]}")
		print(f"Train accuracy: {history.history['accuracy'][best_epoch]}")
		print(f"Validation accuracy: {history.history['val_accuracy'][best_epoch]}")


	def evaluate_model(self) -> list:
		"""
		Evaluate the model.

		Returns
		-------
		list
			Test loss and test accuracy.
		"""
		assert self.model is not None, "Model not defined. Please, build and train the model before evaluating it."
		assert self.trained, "Model not trained. Please, train the model before evaluating."

		result = self.model.evaluate(self.X_test, self.y_test)

		print(f"Test loss: {result[0]}")
		print(f"Test accuracy: {result[1]}")

		return result
	
	def metrics_and_confusion_matrix(self) -> tuple:
		"""
		Plot the confusion matrix of the test set.

		Returns
		-------
		tuple
			average_accuracy, average_precision, average_recall, average_f1_score in the different classes.
		"""
		assert self.model is not None, "Model not defined. Please, build and train the model before evaluating it."
		assert self.trained, "Model not trained. Please, train the model before evaluating."

		y_pred = self.model.predict(self.X_test)
		y_pred_classes = np.argmax(y_pred, axis=1)
		y_true_classes = np.argmax(self.y_test, axis=1)

		cm = confusion_matrix(y_true_classes, y_pred_classes)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(self.num_classes))

		if self.num_classes <= 10:
			fig, ax = plt.subplots(figsize=(10, 10))
		else:
			fig, ax = plt.subplots(figsize=(20, 20))
		disp.plot(ax=ax, cmap='Blues')
		plt.title('Confusion Matrix')
		plt.show()

		# Print metrics (macro)
		average_accuracy = accuracy_score(y_true_classes, y_pred_classes)
		average_precision = precision_score(y_true_classes, y_pred_classes, average='macro')
		average_recall = recall_score(y_true_classes, y_pred_classes, average='macro')
		average_f1_score = f1_score(y_true_classes, y_pred_classes, average='macro')

		print(f"Average accuracy: {average_accuracy}")
		print(f"Average precision (macro): {average_precision}")
		print(f"Average recall (macro): {average_recall}")
		print(f"Average F1-score (macro): {average_f1_score}")

		return average_accuracy, average_precision, average_recall, average_f1_score
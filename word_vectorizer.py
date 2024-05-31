import os
import pickle as pkl
from gensim.models import Word2Vec
from gensim.models import FastText
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import spacy
import warnings
from typing import Literal

nltk.download('punkt')

class WordVectorizer:
	def __init__(self) -> None:
		"""
		Initializes the WordVectorizer class.
		"""
		self.dataset_name: str = ''
		self.size_mb: int|Literal['max'] = 'max'
		self.tokenizer_name: str = ''
		self.caps: bool = False
		self.model_name: str = ''
		self.model: Word2Vec | FastText | None = None
		self.model_type_name: str = ''

		self.raw_texts_path: str = ''
		self.tokenized_texts_path: str = ''
		self.model_path: str = ''

	def load_data(self, dataset: str = 'cat_gc', size_mb: int|Literal['max'] = 100) -> None:
		"""
		Loads the dataset and filters it to a specific size.

		Parameters
		----------
		dataset : str
			The dataset to load. Currently only 'cat_gc' is available.
		size_mb : int|Literal['max']
			The size to filter the dataset to in MB. Common values are 100, 500, 1000.
			If 'max' is selected, the full dataset is loaded.
		"""
		assert dataset in ['cat_gc'], 'Dataset not available.'
		assert size_mb > 0 if isinstance(size_mb, int) else size_mb == 'max', 'Size must be greater than 0 or "max".'

		self.dataset_name = dataset
		self.size_mb = size_mb 
		self.raw_texts_path = f'./data/raw_{dataset}_{size_mb}mb.pkl'

		# Check if the dataset already exists
		if os.path.exists(self.raw_texts_path):
			print(f"Dataset in {self.raw_texts_path} already exists.")
			with open(self.raw_texts_path, 'rb') as f:
				raw_texts = pkl.load(f)
			return
			
		# Load the dataset if it doesn't exist
		raw_dataset = load_dataset(
			"projecte-aina/catalan_general_crawling", 
			split="train", 
			trust_remote_code=True)

		full_raw_texts = raw_dataset["text"]

		# Filter and tokenize the dataset
		if size_mb == 'max':
			raw_texts = full_raw_texts
		else:
			raw_texts = self.__filter_dataset_size(raw_texts=full_raw_texts, size_mb=size_mb)

		# Save the filtered dataset
		with open(self.raw_texts_path, 'wb') as f:
			pkl.dump(raw_texts, f)

	def __filter_dataset_size(self, raw_texts: list[str], size_mb: int) -> list[str]:
		"""
		Filters the dataset to a specific size.

		Parameters
		----------
		raw_texts : list[str]
			The list of texts to process.
		size : int
			The size to filter the dataset to in MB. Common values are 100, 500, 1024 (1GB).

		Returns
		-------
		list[str]
			The first N texts that fit within the size limit.
		"""
		current_size = 0
		return_texts = []
		for text in raw_texts:
			text_size = len(text.encode('utf-8')) / (1024 * 1024)
			if current_size + text_size > size_mb:
				break
			return_texts.append(text)
			current_size += text_size

		return return_texts

	def tokenize(self, tokenizer: str = 'spacy', caps: bool=False, force_tokenize: bool=False) -> None:
		"""
		Tokenizes the texts in the dataset.

		Parameters
		----------
		tokenizer : str
			The tokenizer to use. Options are 'spacy' and 'nltk'.
			If spacy is selected, the 'ca_core_news_sm' model is used.
		caps : bool
			Whether to keep the uppercase characters in the text.
		force_tokenize : bool
			Whether to force the tokenization of the texts even if they are already tokenized.

		Resulting tokenized data is saved to './data' folder.
		"""
		assert tokenizer in ['spacy', 'nltk'], 'Tokenizer not available.'

		file_path = f'./data/{tokenizer}_{self.dataset_name}_{self.size_mb}mb.pkl'
		caps_file_path = f'./data/{tokenizer}_caps_{self.dataset_name}_{self.size_mb}mb.pkl'

		self.tokenizer_name = tokenizer
		self.tokenized_texts_path = caps_file_path if caps else file_path
		self.caps = caps

		# Check if the tokenized texts already exist
		if not force_tokenize and os.path.exists(self.tokenized_texts_path):
			print(f'Tokenized texts in file {caps_file_path if caps else file_path} already exist.')
			return

		# Tokenize the texts if they don't exist
		assert os.path.exists(self.raw_texts_path), 'Raw texts not loaded.'
		with open(self.raw_texts_path, 'rb') as f:
			raw_texts = pkl.load(f)

		count = 0
		total_texts = len(raw_texts)
		tokenized_texts = []

		# Spacy tokenizer
		if tokenizer == 'spacy':
			# Use spacy with cuda if available
			try:
				spacy.require_gpu()
				print("CUDA GPU available. Using spacy with CUDA.")
			except:
				print("CUDA GPU not available. Using spacy without CUDA.")
			
			nlp = spacy.load('ca_core_news_sm')
			
			# We add 1000 to the max_length to avoid problems
			max_texts_lenght = max([len(text) for text in raw_texts])
			nlp.max_length = max_texts_lenght + 1000

			for text in raw_texts:
				tokenized_texts.append([token.text for token in nlp(text)])
				count += 1
				print(f'Tokenized {count+1}/{total_texts} texts with spacy.', end='\r')

		# Nltk tokenizer
		elif tokenizer == 'nltk':
			for text in raw_texts:
				tokenized_texts.append(word_tokenize(text))
				count += 1
				print(f'Tokenized {count+1}/{total_texts} texts with nltk.', end='\r')

		# Save uppercase tokenized texts
		with open(caps_file_path, 'wb') as f:
			pkl.dump(tokenized_texts, f)

		# Save lowercase tokenized texts
		with open(file_path, 'wb') as f:
			pkl.dump([[token.lower() for token in text] for text in tokenized_texts], f)

	def train(self, 
		vectorizer: str='word2vec', 
		model_type: str='skipgram', 
		vector_size: int=100,
		workers: int=1,
		save: bool=True,
		force_train: bool=False
		) -> Word2Vec | FastText:
		"""
		Trains the word vectorizer with the dataset.

		Parameters
		----------
		vectorizer : str
			The vectorizer to use. Options are 'word2vec' and 'fasttext'.
		model_type : str
			The model type to use. Options are 'skipgram' and 'cbow'.
		vector_size : int
			The size of the vectors to use.
		workers : int
			The number of workers to use in the training process (CPU threads).
		save : bool
			Whether to save the model to the './models' folder.
		force_train : bool
			Whether to force the training of the model even if it already exists.

		Returns
		-------
		Word2Vec or FastText
			The object containing the trained model (from gensim).
		"""
		assert vectorizer in ['word2vec', 'fasttext'], 'Vectorizer not available.'
		assert model_type in ['skipgram', 'cbow'], 'Model type not available.'
		assert workers > 0, 'Workers must be greater than 0.'

		assert os.path.exists(self.tokenized_texts_path), 'Tokenized texts not loaded.'
		with open(self.tokenized_texts_path, 'rb') as f:
			tokenized_texts = pkl.load(f)

		self.model_name = vectorizer
		self.model_type_name = model_type
		vectorizer_key_name = 'w2v' if vectorizer == 'word2vec' else 'ft'
		model_type_key_name = 'sg' if model_type == 'skipgram' else 'cbow'
		caps_key_name = '_caps_' if self.caps else '_'
		self.model_path = f'./models/{vectorizer}/{vectorizer_key_name}_{model_type_key_name}_{vector_size}_{self.tokenizer_name}{caps_key_name}{self.dataset_name}_{self.size_mb}mb.model'

		# Check if the model already exists
		if not force_train and os.path.exists(self.model_path):
			print(f'Model in {self.model_path} already exists. Loaded model.')
			if vectorizer == 'word2vec':
				self.model = Word2Vec.load(self.model_path)
			elif vectorizer == 'fasttext':
				self.model = FastText.load(self.model_path)
			return self.model

		# Train the word2vec model
		if vectorizer == 'word2vec':
			self.model = Word2Vec(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=5, workers=workers)

		# Train the fasttext model
		elif vectorizer == 'fasttext':
			self.model = FastText(sentences=tokenized_texts, vector_size=vector_size, window=5, min_count=5, workers=workers)

		if save:
			self.model.save(self.model_path)

		return self.model
		

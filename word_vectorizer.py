import os
import pickle as pkl
from gensim.models import Word2Vec
from gensim.models import FastText
from datasets import load_dataset
import nltk
from nltk.tokenize import word_tokenize
import spacy
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from typing import Literal, Iterator, Any

nltk.download('punkt')

class WordVectorizer:
	def __init__(self, pretrained: bool=False) -> None:
		"""
		Initializes the WordVectorizer class.

		Parameters
		----------
		pretrained : bool
			Whether to model will be pretrained or not.
			If True, the load_data and tokenize methods won't actually load or tokenize the data, 
			but only set the parameters for the pretrained model to use in the train method.
		"""
		assert isinstance(pretrained, bool), 'Pretrained parameter must be a boolean.'

		self.pretrained: bool = pretrained
		self.dataset_name: str = ''
		self.size_mb: int|Literal['max'] = 'max'
		self.total_texts: int = 0
		self.max_texts_lenght: int = 0
		self.tokenizer_name: str = ''
		self.caps: bool = False
		self.model_name: str = ''
		self.model: Word2Vec | FastText | None = None
		self.model_type_name: str = ''

		self.raw_texts_path: str = ''
		self.tokenized_texts_path: str = ''
		self.model_path: str = ''
	
	def __getattr__(self, attr: str) -> Any:
		"""
		Overrides the default __getattr__ method to access the model attributes.
		
		Parameters
		----------
		attr : str
			The attribute name being accessed.
		
		Returns
		-------
		Any
			The attribute from the model if it exists.

		Raises
		------
		AttributeError
			If the attribute does not exist in the model.
		"""
		assert self.model is not None, 'Model not trained.'
		
		try:
			return getattr(self.model, attr)
		except AttributeError:
			raise AttributeError(f"'WordVectorizer' (model of class {self.model.__class__.__name__}) object has no attribute '{attr}'.")

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

		# Check if the model is pretrained (in which case the data is not loaded)
		if self.pretrained:
			return

		# Check if the dataset already exists
		if os.path.exists(self.raw_texts_path):
			print(f"Dataset in {self.raw_texts_path} already exists.")
			with open(self.raw_texts_path, 'rb') as f:
				raw_texts = pkl.load(f)
				self.total_texts = len(raw_texts)
				self.max_texts_lenght = max([len(text) for text in raw_texts])
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
			self.total_texts = len(raw_texts)
			self.max_texts_lenght = max([len(text) for text in raw_texts])

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
		else:
			print(f"Dataset size is less than the specified {size_mb}MB. Loading full dataset.")

		return return_texts

	def tokenize(self, tokenizer: str = 'spacy', caps: bool=False, batch_size: int|Literal['max']='max', force_tokenize: bool=False) -> None:
		"""
		Tokenizes the texts in the dataset.

		Parameters
		----------
		tokenizer : str
			The tokenizer to use. Options are 'spacy' and 'nltk'.
			If spacy is selected, the 'ca_core_news_sm' model is used.
		caps : bool
			Whether to keep the uppercase characters in the text.
		batch_size : int|Literal['max']
			The number of texts to process per batch. If 'max' is selected, all texts are processed at once.
		force_tokenize : bool
			Whether to force the tokenization of the texts even if they are already tokenized.

		Resulting tokenized data is saved to './data' folder.
		"""
		assert tokenizer in ['spacy', 'nltk'], 'Tokenizer not available.'
		assert batch_size > 0 if isinstance(batch_size, int) else batch_size == 'max', 'Batch size must be greater than 0 or "max".'
		assert isinstance(caps, bool), 'Caps must be a boolean.'

		file_path = f'./data/{tokenizer}_{self.dataset_name}_{self.size_mb}mb.pkl'
		caps_file_path = f'./data/{tokenizer}_caps_{self.dataset_name}_{self.size_mb}mb.pkl'

		self.tokenizer_name = tokenizer
		self.tokenized_texts_path = caps_file_path if caps else file_path
		self.caps = caps

		# Check if the model is pretrained (in which case the data is not tokenized)
		if self.pretrained:
			return

		# Check if the tokenized texts already exist
		if (not force_tokenize) and (os.path.exists(self.tokenized_texts_path)):
			print(f'Tokenized texts in file {caps_file_path if caps else file_path} already exist.')
			return
		
		count = 0
		tokenized_texts = []
		
		# In case caps is False, we check if the tokenized texts with caps exist and convert them to lowercase
		if (not force_tokenize) and (not caps) and (os.path.exists(caps_file_path)):
			print(f'Tokenized texts in file {file_path} not found, but uppercase tokenized texts found in {caps_file_path}. Converting to lowercase.')
			for text_batch in self.__batch_iterator(file_path=caps_file_path, batch_size=batch_size if batch_size != 'max' else self.total_texts):
				batch_tokenized = [[token.lower() for token in text] for text in text_batch]
				tokenized_texts.extend(batch_tokenized)
				if batch_size != 'max':
					count += batch_size
					print(f'Converted {count}/{self.total_texts} texts to lowercase.', end='\r')
				else:
					count += self.total_texts
					print(f'Converted {count}/{self.total_texts} texts to lowercase.', end='\r')

			# Save lowercase tokenized texts
			with open(file_path, 'wb') as f:
				pkl.dump(tokenized_texts, f)
			
			return
		
		# Tokenize the texts if they don't exist
		assert os.path.exists(self.raw_texts_path), 'Raw texts not loaded.'
		with open(self.raw_texts_path, 'rb') as f:
			raw_texts = pkl.load(f)

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
			nlp.max_length = self.max_texts_lenght + 1000

			# Iterate through batches and tokenize
			for texts_batch in self.__batch_iterator(file_path=self.raw_texts_path, batch_size=batch_size if batch_size != 'max' else self.total_texts):
				batch_tokenized = [[token.text for token in nlp(text)] for text in texts_batch]
				tokenized_texts.extend(batch_tokenized)
				if batch_size != 'max':
					count += batch_size
					print(f'Tokenized {count}/{self.total_texts} texts with spacy.', end='\r')
				else:
					count += self.total_texts
					print(f'Tokenized {count}/{self.total_texts} texts with spacy.', end='\r')

			# for text in raw_texts:
			# 	tokenized_texts.append([token.text for token in nlp(text)])
			# 	count += 1
			# 	print(f'Tokenized {count+1}/{self.total_texts} texts with spacy.', end='\r')

		# Nltk tokenizer
		elif tokenizer == 'nltk':
			# Iterate through batches and tokenize
			for texts_batch in self.__batch_iterator(file_path=self.raw_texts_path, batch_size=batch_size if batch_size != 'max' else self.total_texts):
				batch_tokenized = [word_tokenize(text) for text in texts_batch]
				tokenized_texts.extend(batch_tokenized)
				if batch_size != 'max':
					count += batch_size
					print(f'Tokenized {count}/{self.total_texts} texts with nltk.', end='\r')
				else:
					count += self.total_texts
					print(f'Tokenized {count}/{self.total_texts} texts with nltk.', end='\r')

			# for text in raw_texts:
			# 	tokenized_texts.append(word_tokenize(text))
			# 	count += 1
			# 	print(f'Tokenized {count+1}/{self.total_texts} texts with nltk.', end='\r')

		# Save uppercase tokenized texts
		with open(caps_file_path, 'wb') as f:
			pkl.dump(tokenized_texts, f)

		# Save lowercase tokenized texts
		with open(file_path, 'wb') as f:
			pkl.dump([[token.lower() for token in text] for text in tokenized_texts], f)

	def __batch_iterator(self, file_path: str, batch_size: int = 1000) -> Iterator[list[str]]:
		"""
		Generator function that yields batches of texts from the dataset.

		Parameters
		----------
		filepath : str
			Path to the file containing the dataset.
		batch_size : int
			Number of texts to process per batch.

		Yields
		------
		Iterator[list[str]]
			Batches of texts.
		"""
		try:
			with open(file_path, 'rb') as file:
				texts = pkl.load(file)
				for i in range(0, len(texts), batch_size):
					yield texts[i:i+batch_size]
		except FileNotFoundError:
			print("File not found. Please check the path and try again.")
			raise

	def train(self, 
		vectorizer: str='word2vec', 
		model_type: str='skipgram', 
		vector_size: int=100,
		window: int=5,
		min_count: int=5,
		workers: int=1,
		save: bool=True,
		force_train: bool=False,
		) -> None:
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
		window : int
			The window size to use in the training process.
		min_count : int
			The minimum count of a word to be included in the training process.
		workers : int
			The number of workers to use in the training process (CPU threads).
		save : bool
			Whether to save the model to the './models' folder.
		force_train : bool
			Whether to force the training of the model even if it already exists.
		"""
		assert vectorizer in ['word2vec', 'fasttext'], 'Vectorizer not available.'
		assert model_type in ['skipgram', 'sg', 'cbow'], 'Model type not available.'
		assert workers > 0, 'Workers must be greater than 0.'
		assert vector_size > 0, 'Vector size must be greater than 0.'
		assert window > 0, 'Window size must be greater than 0.'
		assert min_count > 0, 'Min count must be greater than 0.'

		self.model_name = vectorizer
		self.model_type_name = model_type
		vectorizer_key_name = 'w2v' if vectorizer == 'word2vec' else 'ft'
		model_type_key_name = 'sg' if model_type in ['skipgram', 'sg'] else 'cbow'
		caps_key_name = '_caps_' if self.caps else '_'
		window_key_name = f'_win{window}' if window != 5 else ''
		min_count_key_name = f'minc{min_count}_' if min_count != 5 else ''
		self.model_path = f'./models/{vectorizer}/{vectorizer_key_name}_{model_type_key_name}_{vector_size}{window_key_name}_{min_count_key_name}{self.tokenizer_name}{caps_key_name}{self.dataset_name}_{self.size_mb}mb.model'

		# Check if the model already exists
		if not force_train and os.path.exists(self.model_path):
			print(f'Model in {self.model_path} already exists. Loaded model.')
			if vectorizer == 'word2vec':
				self.model = Word2Vec.load(self.model_path)
			elif vectorizer == 'fasttext':
				self.model = FastText.load(self.model_path)
			return
		
		# Load the tokenized texts
		assert os.path.exists(self.tokenized_texts_path), 'Tokenized texts not loaded.'
		print(f'Loading tokenized texts from {self.tokenized_texts_path}...')
		with open(self.tokenized_texts_path, 'rb') as f:
			tokenized_texts = pkl.load(f)

		# Train the word2vec model
		if vectorizer == 'word2vec':
			print(f'Training {vectorizer} model with {model_type}...')
			self.model = Word2Vec(
				sentences=tokenized_texts, 
				vector_size=vector_size, 
				sg=1 if model_type in ['skipgram', 'sg'] else 0,
				window=window, 
				min_count=min_count, 
				workers=workers)

		# Train the fasttext model
		elif vectorizer == 'fasttext':
			print(f'Training {vectorizer} model with {model_type}...')
			self.model = FastText(
				sentences=tokenized_texts, 
				vector_size=vector_size,
				sg=1 if model_type in ['skipgram', 'sg'] else 0,
				window=window, 
				min_count=min_count, 
				workers=workers)

		if save:
			self.model.save(self.model_path)
	
	def tsne(self, num_words: int=100) -> None:
		"""
		Performs t-SNE on the word vectors and plots the results.

		Parameters
		----------
		num_words : int
			The number of words to plot.
		"""
		assert num_words > 0, 'Number of words must be greater than 0.'
		assert self.model is not None, 'Model not trained.'

		# Get the word vectors sorted by frequency
		word_vectors = self.model.wv.vectors
		word_labels = self.model.wv.index_to_key

		# Perform t-SNE
		tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=5)
		tsne_results = tsne.fit_transform(word_vectors[:num_words])

		# Plot the results
		plt.figure(figsize=(16, 16))
		for i, label in enumerate(word_labels[:num_words]):
			x, y = tsne_results[i, :]
			plt.scatter(x, y)
			plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
		plt.show()
		

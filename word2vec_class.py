from gensim.models import word2vec
from datasets import load_dataset
from nltk.tokenize import word_tokenize, sent_tokenize
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from multiprocessing import Pool, cpu_count
from gensim.models import Word2Vec
import fasttext
import gensim

class TrainWord2Vec:
    def __init__(self, fasttext = False):
        self.data_subset = []
        self.fasttext = fasttext
        if self.fasttext:
            self.train_model = self.train_fasttext_model
        else:
            self.train_model = self.train_word2vec_model
        pass
    def preprocess_data(self, data, size=100,sentences=False):
        if size == "complete":
            size = 10000
        self.size = size
        total_size = 0
        for example in data:
            text = example["text"]
            text_size_mb = len(text.encode('utf-8')) / (1024 * 1024) # Convert bytes to MB
            if total_size + text_size_mb <= size:
                self.data_subset.append(text.lower())
                total_size += text_size_mb
                print(f"Read {total_size}mb from a total of {size}", end="\r")
            else:
                break

        if sentences:
            joined_subset= " ".join(self.data_subset)
            self.tokenized_data = [word_tokenize(sent)  for sent in sent_tokenize(joined_subset)]
        else:
            self.tokenized_data = self.__process_chunks()


    def __tokenize_chunk(self, chunk):
        return [word_tokenize(text.lower()) for text in chunk]
    
    def __process_chunks(self, chunk_size=1000):
        tokenized_data = []
        for i in range(0, len(self.data_subset), chunk_size):
            chunk = self.data_subset[i:i + chunk_size]
            tokenized_data.extend(self.__tokenize_chunk(chunk))
            print(f"Tokenized {i} from {len(self.data_subset)}")
        return tokenized_data
    def save_data(self):
        filename = f"./data/{self.size}mb.txt"
        with open(filename, mode="w", encoding="utf-8") as f:
            for text in self.data_subset:
                f.write(text + "\n")
        return filename

    def train_fasttext_model(self):
        file = self.save_data()
        model = fasttext.train_unsupervised(file)
        model.save_model(f"models/fasttext_{self.size}.bin")
        self.model = gensim.models.fasttext.load_facebook_model(f"models/fasttext_{self.size}.bin")

        
    def train_word2vec_model(self, vector_size = 300, window=5, min_count=5, epochs=25, cbow=False):
        if cbow:
            sg = 0
            name = "CBOW"
        else:
            sg = 1
            name = ""
        self.model = word2vec.Word2Vec(self.tokenized_data, vector_size=vector_size, window=window, min_count=min_count, workers=16, epochs=25, sg=sg)
        self.model.save(f"./models/word2vec{name}_{str(self.size)}_{str(vector_size)}_{str(window)}_{str(min_count)}.model")
    
    def load_model(self, name):
        self.model = word2vec.Word2Vec.load(f"./models/{name}.model")
        
    def evaluate_model(self):
        results = self.model.wv.evaluate_word_pairs('./data/wordsim353.en.ca.txt')
        print(results)
        return results
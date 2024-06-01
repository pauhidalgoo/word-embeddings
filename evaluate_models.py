"""
Script to evaluate the models existing on the './models' folder 
with the model.wv.evaluate_word_pairs('./data/wordsim353.en.ca.txt').
"""

import os
import csv
from gensim.models import Word2Vec, FastText

# Get the list of files ending with '.model' in the './models' folder
w2v_path_list = [os.path.join('./models/word2vec/', f) for f in os.listdir('./models/word2vec/') if f.endswith('.model')]
ft_path_list = [os.path.join('./models/fasttext/', f) for f in os.listdir('./models/fasttext/') if f.endswith('.model')]
num_w2c_models = len(w2v_path_list)
num_ft_models = len(ft_path_list)
num_total_models = num_w2c_models + num_ft_models

# Concatenate the lists
print(f"Found {num_total_models} models (W2V: {num_w2c_models}, FT: {num_ft_models})")

# Load the word similarity dataset
test_path = './data/wordsim353.en.ca.txt'
caps_test_path = './data/wordsim353.en.ca_caps.txt'

# Store the results in a dictionary
results: dict[str, tuple] = {}
counter = 0

# Evaluate Word2Vec models
for model_path in w2v_path_list:
	model = Word2Vec.load(model_path)
	results[model_path] = model.wv.evaluate_word_pairs(test_path)
	counter += 1
	print(f"Evaluated {counter}/{num_total_models} models", end='\r')

# Evaluate FastText models
for model_path in ft_path_list:
	model = FastText.load(model_path)
	results[model_path] = model.wv.evaluate_word_pairs(test_path)
	counter += 1
	print(f"Evaluated {counter}/{num_total_models} models", end='\r')

# Save the results to csv file in the folder './results'
with open('./results/evaluation_results.csv', 'w', newline='') as csvfile:
	writer = csv.writer(csvfile)
	writer.writerow(['Model', 'Pearson', 'Significance', 'OOV ratio'])
	
	for model_path in results.keys():
		spearman, pearson, oov_ratio = results.get(model_path, (None, None, None))
		writer.writerow([model_path, spearman, pearson, oov_ratio])

print(f"Results saved to './results/evaluation_results.csv'.")

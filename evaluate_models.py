"""
Script to evaluate the models existing on the './models' folder 
with the model.wv.evaluate_word_pairs('./data/wordsim353.en.ca.txt').
After evaluating the models, some plots will be generated and saved to the './results/plots' directory.
"""

import os
import pandas as pd
from gensim.models import Word2Vec, FastText
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from functions import calculate_full_dataset_size

################################################################################
# Function to evaluate the models

def evaluate_models(save: bool=True) -> pd.DataFrame:
	"""
	Evaluate the models existing on the './models' folder with the model.wv.evaluate_word_pairs('./data/wordsim353.en.ca.txt').

	Parameters
	----------
	save : bool
		Whether to save the results to a CSV file. The default is True.

	Returns
	-------
	results_df : pd.DataFrame
		The results of the evaluation in a DataFrame sorted by the 'Avg. Statistic' column.
	"""

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
	results_dict: dict[str, dict[str, str|float|None]] = {
		model_path: {
			'Model': os.path.basename(model_path),
			'Avg. Statistic': None, 
			'Pearson': None, 
			'Pearson p-value': None, 
			'Significance': None, 
			'Significance p-value': None, 
			'OOV Ratio': None
			} for model_path in w2v_path_list + ft_path_list
	}

	counter = 0

	# Evaluate Word2Vec models
	for model_path in w2v_path_list:
		caps = 'caps' in model_path
		model = Word2Vec.load(model_path)
		(pearson, p_pearson), (significance, p_significance), oov_ratio = model.wv.evaluate_word_pairs(test_path if not caps else caps_test_path)
		avg_statistic = (pearson + significance) / 2
		
		results_dict[model_path] = {
			'Model': os.path.basename(model_path),
			'Avg. Statistic': avg_statistic, 
			'Pearson': pearson, 
			'Pearson p-value': p_pearson, 
			'Significance': significance, 
			'Significance p-value': p_significance, 
			'OOV Ratio': oov_ratio
		}

		counter += 1
		print(f"Evaluated {counter}/{num_total_models} models with '{test_path}'.", end='\r')

	# Evaluate FastText models
	for model_path in ft_path_list:
		caps = 'caps' in model_path
		model = FastText.load(model_path)
		(pearson, p_pearson), (significance, p_significance), oov_ratio = model.wv.evaluate_word_pairs(test_path if not caps else caps_test_path)
		avg_statistic = (pearson + significance) / 2

		results_dict[model_path] = {
			'Model': os.path.basename(model_path),
			'Avg. Statistic': avg_statistic, 
			'Pearson': pearson, 
			'Pearson p-value': p_pearson, 
			'Significance': significance, 
			'Significance p-value': p_significance, 
			'OOV Ratio': oov_ratio
		}

		counter += 1
		print(f"Evaluated {counter}/{num_total_models} models with '{test_path}'.", end='\r')

	print()

	# Save the results to a CSV file
	results_df = pd.DataFrame(results_dict).T

	# Reorder the columns and sort the values by the 'Avg. Statistic' column
	results_df = results_df[['Model', 'Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio', 'Pearson p-value', 'Significance p-value']]
	results_df = results_df.sort_values(by='Avg. Statistic', ascending=False)

	# Save the results to a CSV file
	if save:
		results_df.to_csv('./results/evaluation_results.csv', index=False)
		print(f"Results saved to './results/evaluation_results.csv'.")

	return results_df

################################################################################
# Functions to generate the plots

# Calculate the size of the full dataset in MB
print("Calculating the size of the full dataset in MB to adjust the scale of the plots...") 
FULL_DATA_SIZE_MB = int(round(calculate_full_dataset_size(), 0))

def generate_model_barplot(results_df: pd.DataFrame, metric: str, family: str='all') -> None:
	"""
	Generate a barplot of the models evaluated for the given metric.

	Parameters
	----------
	results_df : pd.DataFrame
		The DataFrame containing the results of the evaluation.
	metric : str
		The metric to generate the barplot for. It can be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'].
	family : str, optional
		The family of models to generate the barplot for. It can be one of ['all', 'w2v', 'ft']. The default is 'all'.
	"""
	assert metric in ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'], f"Invalid metric '{metric}'. Must be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio']."
	assert family in ['all', 'w2v', 'ft'], f"Invalid family '{family}'. Must be one of ['all', 'w2v', 'ft']."

	# Filter the dataframe by the family of models
	if family == 'w2v':
		results_filtered_df = results_df[results_df['Model'].str.startswith('w2v')].copy()
	elif family == 'ft':
		results_filtered_df = results_df[results_df['Model'].str.startswith('ft')].copy()
	else:
		results_filtered_df = results_df.copy()

	# Sort the values by the metric (descending order, except for 'OOV Ratio')
	results_filtered_df = results_filtered_df.sort_values(by=metric, ascending=(metric == 'OOV Ratio'))

	# Remove .model from the model names
	results_filtered_df['Model'] = results_filtered_df['Model'].apply(lambda x: x.removesuffix('.model'))

	# Generate the barplot
	plt.figure(figsize=(10, 6))
	sns.barplot(x='Model', y=metric, data=results_filtered_df)
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Model')
	plt.ylabel(metric)
	plt.title(f'{metric} for each model ({family})')
	plt.tight_layout()
	plt.savefig(f'./results/plots/bp_model_{family}_{metric.lower().replace(".", "").replace(" ", "_")}.png')
	plt.close()

def generate_tokenizer_barplot(results_df: pd.DataFrame, metric: str, family: str) -> None:
	"""
	Generate a barplot of the average of the given metric for each tokenizer.

	Parameters
	----------
	results_df : pd.DataFrame
		The DataFrame containing the results of the evaluation.
	metric : str
		The metric to generate the barplot for. It can be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'].
	family : str
		The family of models to generate the barplot for. It can be one of ['all', 'w2v', 'ft'].
	"""
	assert metric in ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'], f"Invalid metric '{metric}'. Must be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio']."
	assert family in ['all', 'w2v', 'ft'], f"Invalid family '{family}'. Must be one of ['all', 'w2v', 'ft']."

	# Filter the dataframe by the family of models
	if family == 'w2v':
		results_filtered_df = results_df[results_df['Model'].str.startswith('w2v')].copy()
	elif family == 'ft':
		results_filtered_df = results_df[results_df['Model'].str.startswith('ft')].copy()
	else:
		results_filtered_df = results_df.copy()

	# Extract the tokenizer from the model name ('spacy', 'nltk')
	results_filtered_df['Tokenizer'] = results_filtered_df['Model'].apply(lambda x: 'spacy' if 'spacy' in x else ('nltk' if 'nltk' in x else None))

	# Group by the tokenizer and calculate the average of the metric, ignoring the other columns
	average_tokenizer_df = results_filtered_df[['Tokenizer', metric]].groupby('Tokenizer').mean().reset_index()

	# Generate the barplot
	plt.figure(figsize=(8, 6))
	sns.barplot(x='Tokenizer', y=metric, data=average_tokenizer_df)
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Tokenizer')
	plt.ylabel(metric)
	plt.title(f'{metric} for each tokenizer ({family})')
	plt.tight_layout()
	plt.savefig(f'./results/plots/bp_tokenizer_{family}_{metric.lower().replace(".", "").replace(" ", "_")}.png')
	plt.close()

def generate_size_plot(results_df: pd.DataFrame, metric: str, family: str) -> None:
	"""
	Generate a plot with the evolution of the metric for each dataset size.

	Parameters
	----------
	results_df : pd.DataFrame
		The DataFrame with the evaluation results of the models.
	metric : str
		The metric to generate the barplot for. It can be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'].
	family : str
		The family of models to generate the barplot for. It can be one of ['all', 'w2v', 'ft'].
	"""
	assert metric in ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'], f"Invalid metric '{metric}'. Must be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio']."
	assert family in ['all', 'w2v', 'ft'], f"Invalid family '{family}'. Must be one of ['all', 'w2v', 'ft']."

	# Filter the dataframe by the family of models
	if family == 'w2v':
		results_filtered_df = results_df[results_df['Model'].str.startswith('w2v')].copy()
	elif family == 'ft':
		results_filtered_df = results_df[results_df['Model'].str.startswith('ft')].copy()
	else:
		results_filtered_df = results_df.copy()

	# Extract the size from the model name
	results_filtered_df['Size'] = results_filtered_df['Model'].apply(lambda x: x.split('_')[-1].split('.')[0].removesuffix('mb'))
	results_filtered_df['Size'] = results_filtered_df['Size'].apply(lambda x: int(x) if x != 'max' else FULL_DATA_SIZE_MB)

	results_filtered_df['Model'] = results_filtered_df['Model'].apply(lambda x: '_'.join(x.split('_')[:-1]))

	# Define the ticks and labels for the x-axis
	ticks = results_filtered_df['Size'].unique()
	ticks_labels = []
	for s in ticks:
		if s == FULL_DATA_SIZE_MB:
			ticks_labels.append(f'{s}MB (max)')
		else:
			ticks_labels.append(f'{s}MB')

	# Generate the plot with the evolution of the metric for each dataset size and each model
	plt.figure(figsize=(12, 8))
	sns.lineplot(x='Size', y=metric, hue='Model', data=results_filtered_df, marker='o')
	plt.xticks(ticks=ticks, labels=ticks_labels, rotation=45, ha='right')
	plt.xlabel('Size (MB)')
	plt.ylabel(metric)
	plt.title(f'{metric} for each model and dataset size ({family})')
	plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Model')
	plt.tight_layout()
	plt.savefig(f'./results/plots/lp_size_{family}_{metric.lower().replace(".", "").replace(" ", "_")}.png')
	plt.close()

def generate_vectorizer_barplot(results_df: pd.DataFrame, metric: str, compare: tuple[str, str]=('w2v_sg', 'w2v_cbow')) -> None:
	"""
	Generate a barplot of the average of the given metric for each vectorizer in the comparison (max. 2).

	Parameters
	----------
	results_df : pd.DataFrame
		The DataFrame containing the results of the evaluation.
	metric : str
		The metric to generate the barplot for. It can be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'].
	compare : list[str]|tuple[str]
		The vectorizers to compare. Can be one of ['w2v_sg', 'w2v_cbow', 'ft_sg', 'ft_cbow']. The default is ['w2v', 'ft'].
	"""
	assert metric in ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio'], f"Invalid metric '{metric}'. Must be one of ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio']."

	# Filter the dataframe by the family of models
	results_filtered_df = results_df[results_df['Model'].str.startswith(compare[0]) | results_df['Model'].str.startswith(compare[1])]

	results_filtered_df['Vectorizer'] = results_filtered_df['Model'].apply(lambda x: compare[0] if x.startswith(compare[0]) else (compare[1] if x.startswith(compare[1]) else None))

	# Group by the vectorizer and calculate the average of the metric, ignoring the other columns
	average_vectorizer_df = results_filtered_df[['Vectorizer', metric]].groupby('Vectorizer').mean().reset_index()

	# Generate the barplot
	plt.figure(figsize=(10, 6))
	sns.barplot(x='Vectorizer', y=metric, data=average_vectorizer_df)
	plt.xticks(rotation=45, ha='right')
	plt.xlabel('Vectorizer')
	plt.ylabel(metric)
	plt.title(f'{metric} for each vectorizer ({compare[0]} vs {compare[1]})')
	plt.tight_layout()
	plt.savefig(f'./results/plots/bp_vectorizer_{compare[0]}_vs_{compare[1]}_{metric.lower().replace(".", "").replace(" ", "_")}.png')
	plt.close()

################################################################################
# Main script

def filter_results(df: pd.DataFrame, neg_keyword: str) -> pd.DataFrame:
	"""
	Filter the evaluation results by the negative and positive keywords.

	Parameters
	----------
	df : pd.DataFrame
		The DataFrame with the evaluation results to filter.
	neg_keyword : str
		The negative keyword to filter the results by. All models containing this keyword will be removed.

	Returns
	-------
	results_filtered_df : pd.DataFrame
		The filtered DataFrame with the evaluation results.
	"""
	df = df.copy()
	results_filtered_df = df[~df['Model'].str.contains(neg_keyword)].copy()

	return results_filtered_df

if __name__ == '__main__':
	# Evaluate the models
	results_df = evaluate_models()

	results_df = filter_results(results_df, 'win10') # Comment this line if you want to include the 'win10' models
	# results_df = filter_results(results_df, 'cbow') # Comment this line if you want to include the 'cbow' models

	# Empty the 'plots' directory before generating the plots
	for file in os.listdir('./results/plots/'):
		os.remove(os.path.join('./results/plots/', file))

	# Define the metrics and families
	metrics = ['Avg. Statistic', 'Pearson', 'Significance', 'OOV Ratio']
	different_vectorizers = results_df['Model'].apply(lambda x: '_'.join(x.split('_')[:2])).unique()
	available_families = []
	bool_w2v, bool_ft = False, False
	if any(v.startswith('w2v') for v in different_vectorizers):
		available_families.append('w2v')
		bool_w2v = True
	if any(v.startswith('ft') for v in different_vectorizers):
		available_families.append('ft')
		bool_ft = True

	if bool_w2v and bool_ft:
		available_families.append('all')

	# Generate the plots
	count_plots = 0
	for metric in metrics:
		for family in available_families:
			generate_model_barplot(results_df, metric, family)
			generate_tokenizer_barplot(results_df, metric, family)
			generate_size_plot(results_df, metric, family)

			count_plots += 3
		
		if len(different_vectorizers) > 1:
			for compare in combinations(different_vectorizers, 2):
				generate_vectorizer_barplot(results_df, metric, compare)
				count_plots += 1

	print(f"{count_plots} plots generated from the results. Saved to './results/plots/' directory.")

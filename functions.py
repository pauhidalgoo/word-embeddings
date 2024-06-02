from datasets import load_dataset

def calculate_full_dataset_size(dataset: str = 'cat_gc') -> float:
	"""
	Calculate the size of the full dataset in MB.

	Parameters
	----------
	dataset : str
		Name of the dataset to calculate the size of.

	Returns
	-------
	float
		Size of the dataset in MB.
	"""
	assert dataset in ['cat_gc'], 'Dataset not available.'
	
	if dataset == 'cat_gc':
		raw_dataset = load_dataset(
			"projecte-aina/catalan_general_crawling", 
			split="train", 
			trust_remote_code=True)

		full_raw_texts = raw_dataset["text"]
	
	size = calculate_encoded_size(raw_texts=full_raw_texts)

	print(f"Full '{dataset}' dataset size = {size} MB")

	return size

def calculate_encoded_size(raw_texts: list[str]) -> float:
	encoded_texts = [text.encode('utf-8') for text in raw_texts]
	encoded_size = sum([len(text) for text in encoded_texts])
	encoded_size_MB = encoded_size / (1024 * 1024)

	return encoded_size_MB
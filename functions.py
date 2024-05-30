def filter_dataset_size(texts: list[str], size_mb: int) -> list[str]:
	"""
	Filters the dataset to a specific size.

	Parameters
	----------
	texts : list[str]
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
	for text in texts:
		text_size = len(text.encode('utf-8')) / (1024 * 1024)
		if current_size + text_size > size_mb:
			break
		return_texts.append(text)
		current_size += text_size

	return return_texts

	

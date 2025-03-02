import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	m,n=data.shape
	mean=data.mean(axis=0)
	std= data.std(axis=0)
	standardized_data = (data -mean)  /(std+1e-6)
	min_data=data.min(axis=0)
	max_data =data.max(axis=0)
	normalized_data = (data - min_data)/(max_data-min_data)

	return standardized_data, normalized_data
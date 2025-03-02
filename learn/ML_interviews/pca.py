import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	
	mean=data.mean(axis=0)
	std= data.std(axis=0)

	data= (data-mean)/(std+1e-8)

	cov= np.cov(data,rowvar=False)

	eigenvalues, eigenvectors = np.linalg.eig(cov)

	topk= np.argsort(eigenvalues)[::-1][:k]

	principal_components= eigenvectors[:,topk]

	return np.round(principal_components, 4)
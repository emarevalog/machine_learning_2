import numpy as np

class SVD:
    def __init__(self, n_components=None):
        self.n_components = n_components
        self.s = None
        self.Vt = None

    def fit(self, X):
        covariance_matrix = np.cov(X, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
        sorted_indices = np.argsort(eigen_values)[::-1]
        sorted_eigen_values = eigen_values[sorted_indices]
        sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
        if self.n_components is not None:
            sorted_eigen_values = sorted_eigen_values[:self.n_components]
            sorted_eigen_vectors = sorted_eigen_vectors[:, :self.n_components]
        self.s = np.sqrt(sorted_eigen_values)
        self.Vt = sorted_eigen_vectors.T

    def transform(self, X):
        return np.dot(X, self.Vt.T)

    def fit_transform(self, X):
        self.fit(X)
        return np.dot(X, self.Vt.T) / self.s
import numpy as np

class PCV:
    def __init__(self, n_components=None):
        """
        Initializes the PCV class.

        Parameters:
        - n_components (int, optional): The number of parent components to retain.
        """
        self.n_components = n_components
        self.eigen_values = None
        self.eigen_vectors = None

    def fit(self, X):
        """
        Fits principal component analysis (PCA) to the input data.

        Parameters:
        - X (array-like): The input data, an array of size (m, n).
        """
        # Centrar los datos
        X_meaned = X - np.mean(X, axis=0)
        
        # Calcular la matriz de covarianza
        cov_matrix = np.cov(X_meaned, rowvar=False)
        
        # Calcular los autovalores y autovectores
        self.eigen_values, self.eigen_vectors = np.linalg.eig(cov_matrix)
        
        # Ordenar los autovalores y autovectores de mayor a menor
        idx = self.eigen_values.argsort()[::-1]
        self.eigen_values = self.eigen_values[idx]
        self.eigen_vectors = self.eigen_vectors[:, idx]
        
        # Reducir dimensionalidad si es necesario
        if self.n_components is not None:
            self.eigen_vectors = self.eigen_vectors[:, :self.n_components]

    def fit_transform(self, X):
        """
        Fits principal component analysis (PCA) to the input data
        and transforms the data into the new space.

        Parameters:
        - X (array-like): The input data, an array of size (m, n).

        Returns:
        - X_transformed (array-like): The transformed data in the new space.
        """
        self.fit(X)
        return np.dot(X, self.eigen_vectors)

    def transform(self, X):
        """
        Transforms the data into the space defined by principal component analysis (PCA).

        Parameters:
        - X (array-like): The input data, an array of size (m, n).

        Returns:
        - X_transformed (array-like): The transformed data in the new space.
        """
        return np.dot(X, self.eigen_vectors)

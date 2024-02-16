import numpy as np

class SVD:
    def __init__(self, n_components=None):
        """
        Inicializa la clase SVD.

        Parámetros:
        - n_components (int, opcional): El número de componentes principales a retener.
        """
        self.n_components = n_components
        self.U = None
        self.s = None
        self.Vt = None

    def fit(self, X):
        """
        Ajusta la descomposición de valores singulares (SVD) a los datos de entrada.

        Parámetros:
        - X (array-like): Los datos de entrada, una matriz de tamaño (m, n).
        """
        self.U, self.s, self.Vt = np.linalg.svd(X)
        
        if self.n_components is not None:
            self.U = self.U[:, :self.n_components]
            self.s = self.s[:self.n_components]
            self.Vt = self.Vt[:self.n_components, :]

    def fit_transform(self, X):
        """
        Ajusta la descomposición de valores singulares (SVD) a los datos de entrada
        y transforma los datos en el nuevo espacio.

        Parámetros:
        - X (array-like): Los datos de entrada, una matriz de tamaño (m, n).

        Retorna:
        - X_transformed (array-like): Los datos transformados en el nuevo espacio.
        """
        self.fit(X)
        return np.dot(self.U, np.dot(np.diag(self.s), self.Vt))

    def transform(self, X):
        """
        Transforma los datos en el espacio definido por la descomposición SVD.

        Parámetros:
        - X (array-like): Los datos de entrada, una matriz de tamaño (m, n).

        Retorna:
        - X_transformed (array-like): Los datos transformados en el nuevo espacio.
        """
        return np.dot(X, self.Vt.T)

import numpy as np

class KMedoids:
    def __init__(self, n_clusters=2, max_iter=1000):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    def fit(self, X):
        n_samples = X.shape[0]
        distances = np.zeros((n_samples, n_samples))

        # Calcular la matriz de distancias entre todas las muestras
        for i in range(n_samples):
            for j in range(n_samples):
                distances[i, j] = np.linalg.norm(X[i] - X[j])

        # Inicializar medoids aleatoriamente
        medoid_indices = np.random.choice(n_samples, self.n_clusters, replace=False)

        # Iteraciones del algoritmo
        for _ in range(self.max_iter):
            # Asignar cada punto al medoid más cercano
            labels = np.argmin(distances[medoid_indices], axis=0)

            # Calcular la suma de distancias de cada punto a su medoid correspondiente
            total_distances = np.zeros(n_samples)
            for i in range(n_samples):
                total_distances[i] = distances[i, medoid_indices[labels[i]]]

            # Actualizar los medoids
            for i in range(self.n_clusters):
                cluster_indices = np.where(labels == i)[0]
                cluster_distances = distances[cluster_indices][:, cluster_indices]
                total_distances_cluster = np.sum(cluster_distances, axis=1)
                medoid_index = cluster_indices[np.argmin(total_distances_cluster)]
                medoid_indices[i] = medoid_index

        # Guardar los medoids y las etiquetas
        self.medoid_indices_ = medoid_indices
        self.labels_ = labels

        return self
    
    def transform(self, X):
        """Transforms X to cluster-distance space.

        Parameters
        ----------
        X : numpy.ndarray, shape=(n_query, n_features)
            Data to transform.

        Returns
        -------
        X_new : numpy.ndarray, shape=(n_query, n_clusters)
            X transformed in the new space of distances to cluster centers.
        """
        X = np.asarray(X, dtype=np.float64)
        Y = self.cluster_centers_
        return np.linalg.norm(X[:, np.newaxis, :] - Y[np.newaxis, :, :], axis=-1)

    def predict(self, X):
        """Predicts the closest cluster for each sample in X.

        Parameters
        ----------
        X : numpy.ndarray, shape=(n_query, n_features)
            New data to predict.

        Returns
        -------
        numpy.ndarray, shape=(n_query,)
            Index of the cluster each sample belongs to.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.argmin(self.transform(X), axis=1)

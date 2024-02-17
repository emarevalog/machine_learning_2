import numpy as np

class KMedoids:
    def __init__(self, n_clusters, max_iters=1000, tol=1e-5):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.tol = tol
        self.medoids = None
        self.labels = None
        
    def fit(self, X):
        n_samples, _ = X.shape
        # Initialize medoids randomly
        self.medoids = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        for _ in range(self.max_iters):
            # Assign each data point to the nearest medoid
            distances = self._calc_distances(X)
            labels = np.argmin(distances, axis=1)
            
            # Update medoids
            new_medoids = np.copy(self.medoids)
            for i in range(self.n_clusters):
                cluster_indices = np.where(labels == i)[0]
                cluster_distances = distances[cluster_indices][:, cluster_indices]
                total_distances = np.sum(cluster_distances, axis=1)
                medoid_index = cluster_indices[np.argmin(total_distances)]
                new_medoids[i] = medoid_index
            
            # Check for convergence
            if np.array_equal(new_medoids, self.medoids):
                break
                
            self.medoids = new_medoids
        
        self.labels = np.argmin(distances, axis=1)
            
    def predict(self, X):
        distances = self._calc_distances(X)
        return np.argmin(distances, axis=1)
        
    def _calc_distances(self, X):
        distances = np.zeros((X.shape[0], self.n_clusters))
        for i, medoid_index in enumerate(self.medoids):
            medoid = X[medoid_index]
            distances[:, i] = np.linalg.norm(X - medoid, axis=1)
        return distances

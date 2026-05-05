from abc import ABC, abstractmethod
import numpy as np
import hdbscan
import umap
from river import cluster

class ClusteringWrapper(ABC):
    @abstractmethod
    def fit_predict(self, X_train, X_test, kwargs):
        """
        Takes training data and test data.
        Returns cluster assignments for X_test as a 1D numpy array.
        Noise points should be assigned to -1.
        """
        pass

class StreamKMeansWrapper(ClusteringWrapper):
    def fit_predict(self, X_train, X_test, kwargs):
        model = cluster.STREAMKMeans(
            n_clusters=kwargs.get('n_clusters', 10),
            halflife=kwargs.get('halflife', 0.5),
            sigma=kwargs.get('sigma', 1.0),
            seed=kwargs.get('seed', 42)
        )

        # Fit on train stream
        for x in X_train:
            model = model.learn_one({i: v for i, v in enumerate(x)})

        # Predict on test stream
        preds = []
        for x in X_test:
            pred = model.predict_one({i: v for i, v in enumerate(x)})
            # If river returns a dictionary of clusters or an int
            if pred is None:
                preds.append(-1)
            else:
                preds.append(pred)
                # optionally learn on test? No, usually evaluate on test frozen.

        return np.array(preds)

class DenStreamWrapper(ClusteringWrapper):
    def fit_predict(self, X_train, X_test, kwargs):
        model = cluster.DenStream(
            epsilon=kwargs.get('epsilon', 0.5),
            beta=kwargs.get('beta', 0.2),
            mu=kwargs.get('mu', 2.0)
        )

        # Fit on train stream
        for x in X_train:
            model = model.learn_one({i: v for i, v in enumerate(x)})

        preds = []
        for x in X_test:
            pred = model.predict_one({i: v for i, v in enumerate(x)})
            if pred is None:
                preds.append(-1) # Noise
            else:
                preds.append(pred)

        return np.array(preds)

class HDBSCANWrapper(ClusteringWrapper):
    def fit_predict(self, X_train, X_test, kwargs):
        model = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 5),
            min_samples=kwargs.get('min_samples', None),
            core_dist_n_jobs=1 # enforce single-thread for timing
        )

        # HDBSCAN is batch. Standard practice for batch clustering is to fit on test,
        # OR fit on train and use approximate_predict. We'll cluster the test set directly.
        # This matches the prompt: "For batch methods, fit_predict on the test set".
        preds = model.fit_predict(X_test)
        return preds

class UMAPHDBSCANWrapper(ClusteringWrapper):
    def fit_predict(self, X_train, X_test, kwargs):
        reducer = umap.UMAP(
            n_neighbors=kwargs.get('n_neighbors', 15),
            min_dist=kwargs.get('min_dist', 0.1),
            n_components=10, # Standard requirement for embedding reduction
            random_state=kwargs.get('seed', 42),
            n_jobs=1
        )

        # Fit UMAP on train, transform test
        if len(X_train) > reducer.n_neighbors:
            reducer.fit(X_train)
            X_test_reduced = reducer.transform(X_test)
        else:
            # Fallback if train is too small (e.g. in mock data)
            X_test_reduced = reducer.fit_transform(X_test)

        model = hdbscan.HDBSCAN(
            min_cluster_size=kwargs.get('min_cluster_size', 5),
            min_samples=kwargs.get('min_samples', None),
            core_dist_n_jobs=1
        )

        preds = model.fit_predict(X_test_reduced)
        return preds

def get_algorithm(algo_name: str) -> ClusteringWrapper:
    if algo_name == "stream_kmeans":
        return StreamKMeansWrapper()
    elif algo_name == "denstream":
        return DenStreamWrapper()
    elif algo_name == "hdbscan":
        return HDBSCANWrapper()
    elif algo_name == "umap_hdbscan":
        return UMAPHDBSCANWrapper()
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

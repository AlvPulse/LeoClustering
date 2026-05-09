import numpy as np
from abc import ABC, abstractmethod
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.pairwise import cosine_similarity
import scipy.spatial.distance

class Scorer(ABC):
    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """
        Takes an array of shape (n_samples, dim) and returns a 1D array of scores.
        Higher score = more likely to be the target class.
        """
        pass

class PrototypeScorer(Scorer):
    def __init__(self, prototype):
        self.prototype = prototype

    def score(self, X: np.ndarray) -> np.ndarray:
        return cosine_similarity(X, self.prototype.reshape(1, -1)).flatten()

class KNNScorer(Scorer):
    def __init__(self, exemplars, k):
        self.exemplars = exemplars
        self.k = min(k, len(exemplars))
        # Note: Scikit-learn NearestNeighbors is picklable.
        self.nn = NearestNeighbors(n_neighbors=self.k, metric='cosine')
        self.nn.fit(self.exemplars)

    def score(self, X: np.ndarray) -> np.ndarray:
        # Distance output from cosine metric in sklearn: 1 - cosine_similarity
        # Lower distance = higher similarity
        dist, _ = self.nn.kneighbors(X)
        # Average cosine similarity across the k nearest
        sims = 1.0 - dist
        return np.mean(sims, axis=1)

class MahalanobisScorer(Scorer):
    def __init__(self, mean, cov_inv, pca):
        self.mean = mean
        self.cov_inv = cov_inv
        self.pca = pca

    def score(self, X: np.ndarray) -> np.ndarray:
        if self.pca is not None:
            X = self.pca.transform(X)

        diff = X - self.mean
        # negative mahalanobis distance: Higher is better
        dist_sq = np.sum(np.dot(diff, self.cov_inv) * diff, axis=1)
        return -np.sqrt(dist_sq)

class OCSVMScorer(Scorer):
    def __init__(self, model):
        self.model = model

    def score(self, X: np.ndarray) -> np.ndarray:
        # decision_function: strictly > 0 for inliers in sklearn OCSVM
        # Higher score = more likely inlier
        return self.model.decision_function(X)

class LogisticRegressionScorer(Scorer):
    def __init__(self, model):
        self.model = model

    def score(self, X: np.ndarray) -> np.ndarray:
        # probability of class 1
        return self.model.predict_proba(X)[:, 1]

class FewShotFitter:
    def __init__(self, method_name, config):
        self.method_name = method_name
        self.config = config

    def fit(self, exemplars: np.ndarray, train_pool_all_X: np.ndarray, train_pool_all_y: np.ndarray, target_class: str) -> Scorer:
        n = len(exemplars)

        if self.method_name == "prototype":
            # L2 normalized mean
            proto = np.mean(exemplars, axis=0)
            norm = np.linalg.norm(proto)
            if norm > 0: proto = proto / norm
            return PrototypeScorer(proto)

        elif self.method_name.startswith("knn"):
            k = self.config.get("n_neighbors", 1)
            return KNNScorer(exemplars, k)

        elif self.method_name == "mahalanobis":
            if n < 2:
                raise ValueError("N/A — N too small for covariance estimation")

            dim = exemplars.shape[1]
            max_components = self.config.get("max_pca_components", 64)
            n_components = min(n - 1, max_components)

            pca = None
            if dim > n_components:
                # Fit PCA class-agnostic on entire training split
                pca = PCA(n_components=n_components)
                pca.fit(train_pool_all_X)
                ex_reduced = pca.transform(exemplars)
            else:
                ex_reduced = exemplars

            mean = np.mean(ex_reduced, axis=0)
            lw = LedoitWolf().fit(ex_reduced)
            cov_inv = lw.precision_

            return MahalanobisScorer(mean, cov_inv, pca)

        elif self.method_name == "ocsvm":
            model = OneClassSVM(kernel=self.config.get("kernel", "rbf"), gamma=self.config.get("gamma", "scale"))
            model.fit(exemplars)
            return OCSVMScorer(model)

        elif self.method_name == "logistic_regression":
            # Sample negatives
            neg_mask = train_pool_all_y != target_class
            neg_X = train_pool_all_X[neg_mask]

            neg_count = self.config.get("negative_ratio", 10) * n
            if len(neg_X) > neg_count:
                idx = np.random.choice(len(neg_X), neg_count, replace=False)
                neg_X = neg_X[idx]
                actual_negatives = neg_count
            else:
                actual_negatives = len(neg_X)
                # We will log a warning during the benchmark loop

            X_train = np.vstack([exemplars, neg_X])
            y_train = np.array([1]*n + [0]*len(neg_X))

            model = LogisticRegression(C=self.config.get("c_reg", 1.0), max_iter=self.config.get("max_iter", 1000))
            if len(np.unique(y_train)) < 2:
                raise ValueError("N/A - Not enough negative classes available in training pool.")

            model.fit(X_train, y_train)
            scorer = LogisticRegressionScorer(model)
            scorer.actual_negatives = actual_negatives
            return scorer

        else:
            raise ValueError(f"Unknown method {self.method_name}")

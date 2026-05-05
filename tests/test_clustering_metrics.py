import numpy as np
import pytest
from src.clustering.metrics import evaluate_clustering

def test_perfect_clustering():
    y_true = np.array(["a", "a", "b", "b", "rare", "rare"])
    y_pred = np.array([0, 0, 1, 1, 2, 2])
    rare_classes = ["rare"]

    res = evaluate_clustering(y_true, y_pred, rare_classes)

    assert res["ari"] == 1.0
    assert res["nmi"] == 1.0
    assert res["weighted_purity"] == 1.0
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["survival_flag"] is True
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["purity"] == 1.0
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["noise_rate"] == 0.0

def test_all_noise():
    y_true = np.array(["a", "a", "rare"])
    y_pred = np.array([-1, -1, -1])
    rare_classes = ["rare"]

    res = evaluate_clustering(y_true, y_pred, rare_classes)

    assert res["n_clusters"] == 0
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["survival_flag"] is False
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["noise_rate"] == 1.0

def test_single_cluster_failure():
    y_true = np.array(["a", "a", "a", "a", "rare"])
    y_pred = np.array([0, 0, 0, 0, 0])
    rare_classes = ["rare"]

    res = evaluate_clustering(y_true, y_pred, rare_classes)

    assert res["n_clusters"] == 1
    assert res["largest_cluster_fraction"] == 1.0

    # Rare class is absorbed into 'a', so it did not survive
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["survival_flag"] is False
    assert res["rare_metrics"]["rare_class_metrics"]["rare"]["purity"] == 0.2

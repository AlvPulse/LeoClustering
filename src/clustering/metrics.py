import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

def compute_cluster_purity(y_true, y_pred):
    """
    Compute cluster purity, weighted by cluster size.
    Ignores noise (-1).
    """
    valid_mask = y_pred != -1
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    if len(y_pred_valid) == 0:
        return 0.0

    clusters = np.unique(y_pred_valid)
    majority_sum = 0
    for c in clusters:
        c_mask = y_pred_valid == c
        c_labels = y_true_valid[c_mask]
        if len(c_labels) > 0:
            majority_count = np.max(np.unique(c_labels, return_counts=True)[1])
            majority_sum += majority_count

    return majority_sum / len(y_true_valid)

def compute_rare_class_metrics(y_true, y_pred, rare_classes):
    """
    Computes per-class recall, survival flag, rare-class purity, and noise absorption rate.
    y_pred = -1 denotes noise.
    """
    metrics = {}

    unique_classes = np.unique(y_true)
    per_class_recall = {}

    for cls in unique_classes:
        cls_mask = y_true == cls
        cls_preds = y_pred[cls_mask]

        if len(cls_preds) == 0:
            per_class_recall[cls] = 0.0
            continue

        # Ignore noise for recall to a cluster
        cls_preds_valid = cls_preds[cls_preds != -1]

        if len(cls_preds_valid) == 0:
            per_class_recall[cls] = 0.0
        else:
            # Find the dominant cluster for this class
            vals, counts = np.unique(cls_preds_valid, return_counts=True)
            dominant_cluster = vals[np.argmax(counts)]
            dominant_count = np.max(counts)
            per_class_recall[cls] = dominant_count / len(cls_preds)

    metrics['per_class_recall'] = per_class_recall

    # Rare class specifics
    rare_metrics = {}
    for rc in rare_classes:
        rc_mask = y_true == rc
        rc_preds = y_pred[rc_mask]

        if len(rc_preds) == 0:
            rare_metrics[rc] = {
                "survival_flag": False,
                "purity": 0.0,
                "noise_rate": 0.0
            }
            continue

        noise_rate = np.sum(rc_preds == -1) / len(rc_preds)

        rc_preds_valid = rc_preds[rc_preds != -1]
        if len(rc_preds_valid) == 0:
            rare_metrics[rc] = {
                "survival_flag": False,
                "purity": 0.0,
                "noise_rate": noise_rate
            }
            continue

        vals, counts = np.unique(rc_preds_valid, return_counts=True)
        dominant_cluster = vals[np.argmax(counts)]

        # Check if the dominant cluster actually has this rare class as its plurality label
        dom_cluster_mask = y_pred == dominant_cluster
        dom_cluster_labels = y_true[dom_cluster_mask]
        dom_vals, dom_counts = np.unique(dom_cluster_labels, return_counts=True)
        cluster_plurality_label = dom_vals[np.argmax(dom_counts)]

        survival_flag = (cluster_plurality_label == rc)

        # Rare class purity: fraction of the dominant cluster that is actually the rare class
        purity = np.sum(dom_cluster_labels == rc) / len(dom_cluster_labels)

        rare_metrics[rc] = {
            "survival_flag": survival_flag,
            "purity": purity,
            "noise_rate": noise_rate
        }

    metrics['rare_class_metrics'] = rare_metrics
    return metrics

def evaluate_clustering(y_true, y_pred, rare_classes):
    n_clusters = len(np.unique(y_pred[y_pred != -1]))

    if len(y_pred) == 0 or n_clusters == 0:
        return {
            "ari": 0.0,
            "nmi": 0.0,
            "weighted_purity": 0.0,
            "n_clusters": 0,
            "largest_cluster_fraction": 0.0,
            "rare_metrics": compute_rare_class_metrics(y_true, y_pred, rare_classes)
        }

    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    purity = compute_cluster_purity(y_true, y_pred)

    vals, counts = np.unique(y_pred[y_pred != -1], return_counts=True)
    largest_frac = np.max(counts) / len(y_pred) if len(counts) > 0 else 0.0

    return {
        "ari": ari,
        "nmi": nmi,
        "weighted_purity": purity,
        "n_clusters": n_clusters,
        "largest_cluster_fraction": largest_frac,
        "rare_metrics": compute_rare_class_metrics(y_true, y_pred, rare_classes)
    }

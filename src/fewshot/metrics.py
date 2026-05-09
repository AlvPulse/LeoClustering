import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

def compute_precision_at_recall(precisions, recalls, target_recall):
    """
    Finds the maximum precision where recall >= target_recall.
    """
    valid_mask = recalls >= target_recall
    if not np.any(valid_mask):
        return 0.0
    return np.max(precisions[valid_mask])

def compute_recall_at_fpr(y_true, scores, target_fpr):
    """
    Computes recall at a specific False Positive Rate.
    If there are not enough negative samples to measure the target_fpr (i.e. 1 FP means a higher FPR than target),
    returns None.

    y_true: binary labels (1=positive, 0=negative)
    """
    neg_count = np.sum(y_true == 0)

    # We require at least floor(1/target_fpr) negatives to reliably measure exactly that FPR
    # Actually, we can measure 0 FPs at any negative count, but mathematically if neg_count < 1/target_fpr,
    # the smallest non-zero FPR we can measure is > target_fpr.
    # Prompt asks: "If the test set is too small to evaluate at FPR=1/day, report N/A".
    if neg_count < (1.0 / target_fpr):
        return None

    desc_idx = np.argsort(scores)[::-1]
    y_true_sorted = y_true[desc_idx]

    fps = np.cumsum(y_true_sorted == 0)
    tps = np.cumsum(y_true_sorted == 1)

    max_fps = int(np.floor(target_fpr * neg_count))

    if max_fps == 0:
        valid_mask = fps == 0
    else:
        valid_mask = fps <= max_fps

    if not np.any(valid_mask):
        return 0.0

    total_positives = np.sum(y_true == 1)
    if total_positives == 0:
        return 0.0

    max_tps = np.max(tps[valid_mask])
    return max_tps / total_positives

def evaluate_fewshot_metrics(y_true, scores):
    """
    Evaluates scoring output against ground truth.
    """
    if np.sum(y_true) == 0:
        raise ValueError("No positive examples in test set.")
    if np.sum(y_true == 0) == 0:
        raise ValueError("No negative examples in test set.")

    ap = average_precision_score(y_true, scores)
    precisions, recalls, _ = precision_recall_curve(y_true, scores)

    p_at_50 = compute_precision_at_recall(precisions, recalls, 0.5)
    p_at_80 = compute_precision_at_recall(precisions, recalls, 0.8)
    p_at_95 = compute_precision_at_recall(precisions, recalls, 0.95)

    fpr_hour = 1.0 / 3600.0
    fpr_day = 1.0 / 86400.0

    r_at_fpr_hour = compute_recall_at_fpr(y_true, scores, fpr_hour)
    r_at_fpr_day = compute_recall_at_fpr(y_true, scores, fpr_day)

    return {
        "average_precision": ap,
        "precision_at_recall_0.5": p_at_50,
        "precision_at_recall_0.8": p_at_80,
        "precision_at_recall_0.95": p_at_95,
        "recall_at_fpr_1_hour": r_at_fpr_hour,
        "recall_at_fpr_1_day": r_at_fpr_day
    }

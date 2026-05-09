import numpy as np
import pytest
from src.fewshot.metrics import evaluate_fewshot_metrics, compute_recall_at_fpr

def test_ap_and_precisions():
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0])

    metrics = evaluate_fewshot_metrics(y_true, scores)
    assert metrics["average_precision"] == 1.0
    assert metrics["precision_at_recall_0.5"] == 1.0
    assert metrics["precision_at_recall_0.95"] == 1.0

def test_recall_at_fpr():
    # 5 positives, 86400 negatives to satisfy 1/day constraint
    y_true = np.array([1]*5 + [0]*86400)

    scores = np.zeros(86405)

    scores[0:3] = 10.0
    scores[5:7] = 9.0
    scores[3:5] = 8.0

    # Sort order of scores:
    # 10.0 -> idx 0,1,2 (Positives) -> 3 TP, 0 FP
    # 9.0  -> idx 5,6 (Negatives) -> 3 TP, 2 FP
    # 8.0  -> idx 3,4 (Positives) -> 5 TP, 2 FP

    # FPR = 1/hour = 1/3600. Max FP out of 86400 = floor(86400/3600) = 24
    # We only have 2 FPs at score >= 8.0. So 2 <= 24. We capture all 5 positives!
    # Therefore, Recall at FPR=1/hour should be 1.0 (5/5).
    metrics = evaluate_fewshot_metrics(y_true, scores)
    assert metrics["recall_at_fpr_1_hour"] == 1.0

    # For FPR=1/day = 1/86400. Max FP out of 86400 = 1
    # At score >= 8.0 we have 2 FPs (which is > 1). Thus invalid.
    # At score >= 9.0 we have 2 FPs (which is > 1). Thus invalid.
    # At score >= 10.0 we have 0 FPs (which is <= 1). Valid!
    # At score >= 10.0 we have 3 TPs.
    # Therefore, Recall should be 3/5 = 0.6
    assert metrics["recall_at_fpr_1_day"] == 0.6

def test_recall_at_fpr_direct():
    y_true = np.array([1]*5 + [0]*86400)
    scores = np.zeros(86405)
    scores[0:3] = 10.0
    scores[5:7] = 9.0
    scores[3:5] = 8.0

    fpr_hour = 1.0 / 3600.0
    fpr_day = 1.0 / 86400.0

    assert compute_recall_at_fpr(y_true, scores, fpr_hour) == 1.0
    assert compute_recall_at_fpr(y_true, scores, fpr_day) == 0.6

def test_too_small_test_set():
    y_true = np.array([1, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6])

    metrics = evaluate_fewshot_metrics(y_true, scores)

    assert metrics["recall_at_fpr_1_hour"] is None
    assert metrics["recall_at_fpr_1_day"] is None

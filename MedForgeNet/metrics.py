from __future__ import annotations

from typing import Dict, Union

import numpy as np


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(-scores)
    ranked = labels[order].astype(np.int64)
    positives = ranked == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return float("nan")
    true_pos = np.cumsum(positives)
    ranks = np.arange(1, len(ranked) + 1)
    precision = true_pos / ranks
    return float(precision[positives].sum() / n_pos)


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float = 0.5) -> Dict[str, Union[float, int]]:
    result: Dict[str, Union[float, int]] = {
        "ap": float("nan"),
        "acc_fake": float("nan"),
        "acc_real": float("nan"),
        "avg_acc": float("nan"),
        "n": int(len(labels)),
        "mean_score_real": float("nan"),
        "mean_score_fake": float("nan"),
    }
    if len(labels) == 0:
        return result
    preds = (scores >= threshold).astype(np.int64)
    real_mask = labels == 0
    fake_mask = labels == 1
    if real_mask.any():
        result["acc_real"] = float((preds[real_mask] == 0).mean())
        result["mean_score_real"] = float(scores[real_mask].mean())
    if fake_mask.any():
        result["acc_fake"] = float((preds[fake_mask] == 1).mean())
        result["mean_score_fake"] = float(scores[fake_mask].mean())
    if len(np.unique(labels)) >= 2:
        result["ap"] = average_precision(labels, scores)
        result["avg_acc"] = float((result["acc_real"] + result["acc_fake"]) / 2.0)
    return result

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_squared_error,
    r2_score,
    confusion_matrix,
)


@dataclass(frozen=True)
class MetricPack:
    metrics: Dict[str, Any]
    confusion: Optional[Dict[str, Any]] = None


def classification_metrics(y_true, y_pred) -> MetricPack:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred)
    return MetricPack(
        metrics={
            "accuracy": acc,
            "f1_macro": f1,
        },
        confusion={
            "matrix": cm.tolist(),
        },
    )


def regression_metrics(y_true, y_pred) -> MetricPack:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return MetricPack(
        metrics={
            "rmse": rmse,
            "r2": r2,
        },
        confusion=None,
    )


def summarize_cv(scores: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
    }

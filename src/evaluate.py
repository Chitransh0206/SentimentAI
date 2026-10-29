"""Metric computation for a fitted classifier."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from config import settings


def compute_metrics(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str] = settings.LABELS
) -> dict:
    """Return a JSON-serialisable metrics dictionary.

    ``zero_division=0`` is used so a class that is never predicted gets precision 0
    instead of raising warnings or NaN.
    """
    labels = list(labels)
    out: dict = {"n_samples": int(len(y_true)), "labels": labels, "accuracy": float(accuracy_score(y_true, y_pred))}
    for avg in ("macro", "weighted"):
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        out[f"precision_{avg}"], out[f"recall_{avg}"], out[f"f1_{avg}"] = float(p), float(r), float(f)

    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out["per_class"] = {
        label: {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(support[i]),
        }
        for i, label in enumerate(labels)
    }
    out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=labels).astype(int).tolist()
    out["classification_report"] = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    out["classification_report_text"] = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, digits=3
    )
    return out


def mean_and_std(values: Sequence[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std())

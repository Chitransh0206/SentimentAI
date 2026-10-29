"""Reproducible training: model comparison with cross-validation, final test evaluation.

Leakage protection
------------------
1. The data is split into train/test ONCE, stratified, before anything is fitted.
2. Candidate models are compared by cross-validation on the TRAIN split only. Each
   fold clones the whole Pipeline, so the TF-IDF vocabulary is re-fitted inside
   every fold and never sees the validation fold.
3. The winner is re-fitted on the full train split and scored on the untouched
   test split exactly once.
"""
from __future__ import annotations

import logging
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import settings
from src.data_loader import (
    CleaningReport,
    class_balance_warnings,
    load_dataset,
    validate_class_sufficiency,
)
from src.evaluate import compute_metrics, mean_and_std
from src.model_utils import save_artifacts
from src.preprocessing import TOKEN_PATTERN, normalize_text

logger = logging.getLogger(__name__)


def make_vectorizer() -> TfidfVectorizer:
    """TF-IDF over unigrams+bigrams. No stop-word removal so negations survive."""
    return TfidfVectorizer(
        preprocessor=normalize_text,
        token_pattern=TOKEN_PATTERN,
        lowercase=False,  # normalize_text already lower-cases
        ngram_range=(1, 2),  # bigrams capture "not good", "never again"
        min_df=2,  # ignore one-off tokens (typos, names)
        max_features=200_000,
        sublinear_tf=True,  # dampen very frequent terms
        dtype=np.float32,
    )


def build_candidates(seed: int = settings.RANDOM_SEED) -> dict[str, Pipeline]:
    """The three candidate pipelines. All expose ``predict_proba``."""
    return {
        "Multinomial Naive Bayes": Pipeline(
            [("tfidf", make_vectorizer()), ("clf", MultinomialNB(alpha=0.5))]
        ),
        "Logistic Regression": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                ("clf", LogisticRegression(C=2.0, max_iter=1000, class_weight="balanced", random_state=seed)),
            ]
        ),
        # LinearSVC has no probabilities, so it is wrapped in a sigmoid calibrator.
        # The calibrator is fitted by internal cross-validation on training data only.
        "Linear SVM (calibrated)": Pipeline(
            [
                ("tfidf", make_vectorizer()),
                (
                    "clf",
                    CalibratedClassifierCV(
                        estimator=LinearSVC(C=0.5, class_weight="balanced", random_state=seed),
                        method="sigmoid",
                        cv=3,
                    ),
                ),
            ]
        ),
    }


@dataclass
class TrainingResult:
    pipeline: Pipeline
    metrics: dict
    metadata: dict
    train_index: list[int]
    test_index: list[int]


def split_data(df: pd.DataFrame, test_size: float, seed: int):
    """Stratified train/test split; returns the two frames."""
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df["sentiment"]
    )
    return train_df, test_df


def compare_candidates(train_df: pd.DataFrame, seed: int, folds: int = settings.CV_FOLDS) -> list[dict]:
    """Cross-validate every candidate on the training split only."""
    min_class = int(train_df["sentiment"].value_counts().min())
    n_splits = max(2, min(folds, min_class))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rows = []
    for name, pipeline in build_candidates(seed).items():
        logger.info("Cross-validating %s (%d folds)...", name, n_splits)
        scores = cross_validate(
            pipeline, train_df["text"], train_df["sentiment"], cv=cv,
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy"}, n_jobs=1,
        )
        f1_mean, f1_std = mean_and_std(scores["test_f1_macro"])
        acc_mean, acc_std = mean_and_std(scores["test_accuracy"])
        rows.append({
            "model": name, "cv_folds": n_splits,
            "cv_f1_macro_mean": f1_mean, "cv_f1_macro_std": f1_std,
            "cv_accuracy_mean": acc_mean, "cv_accuracy_std": acc_std,
        })
    return rows


def train_and_evaluate(
    df: pd.DataFrame,
    cleaning_report: CleaningReport | None = None,
    dataset_name: str = "in-memory",
    seed: int = settings.RANDOM_SEED,
    test_size: float = settings.TEST_SIZE,
) -> TrainingResult:
    """Run the whole selection + evaluation procedure on a clean canonical frame."""
    counts = {k: int(v) for k, v in df["sentiment"].value_counts().items()}
    validate_class_sufficiency(counts)
    warnings = class_balance_warnings(counts)
    for w in warnings:
        logger.warning(w)

    train_df, test_df = split_data(df, test_size, seed)
    comparison = compare_candidates(train_df, seed)
    # Selection: highest mean CV macro F1; ties resolved by the earlier (simpler) candidate.
    best = max(comparison, key=lambda r: r["cv_f1_macro_mean"])
    best_name = best["model"]
    logger.info("Selected %s (CV macro F1 %.4f)", best_name, best["cv_f1_macro_mean"])

    final = build_candidates(seed)[best_name]
    final.fit(train_df["text"], train_df["sentiment"])
    test_pred = final.predict(test_df["text"])
    test_metrics = compute_metrics(test_df["sentiment"].tolist(), list(test_pred))

    is_demo = dataset_name.lower().startswith("demo")
    if is_demo:
        warnings.append("Trained on the synthetic demonstration dataset: metrics only prove the software runs.")
    clf = final.named_steps["clf"]
    metrics = {
        "selected_model": best_name,
        "selection_metric": "mean cross-validated macro F1 (training split only)",
        "validation": comparison,
        "test": test_metrics,
    }
    metadata = {
        "model_name": best_name,
        "classifier_repr": repr(clf),
        "classes": list(final.classes_),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_seed": seed,
        "test_size": test_size,
        "dataset_name": dataset_name,
        "is_demonstration_data": is_demo,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "class_counts": counts,
        "train_class_counts": {k: int(v) for k, v in train_df["sentiment"].value_counts().items()},
        "test_class_counts": {k: int(v) for k, v in test_df["sentiment"].value_counts().items()},
        "vocabulary_size": int(len(final.named_steps["tfidf"].vocabulary_)),
        "cleaning_report": cleaning_report.to_dict() if cleaning_report else None,
        "warnings": warnings,
        "versions": {
            "python": platform.python_version(), "scikit_learn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__,
        },
    }
    return TrainingResult(final, metrics, metadata, list(train_df.index), list(test_df.index))


def run_training(
    dataset_path: str | Path | None = None,
    text_column: str = settings.DEFAULT_TEXT_COLUMN,
    label_column: str = settings.DEFAULT_LABEL_COLUMN,
    label_map: Mapping[str, str] | None = None,
    drop_unsupported_labels: bool = False,
    seed: int = settings.RANDOM_SEED,
    test_size: float = settings.TEST_SIZE,
    save: bool = True,
) -> TrainingResult:
    """Load -> clean -> compare -> evaluate -> save. Used by ``scripts/train_model.py``."""
    path = Path(dataset_path) if dataset_path else settings.default_dataset_path()
    df, report = load_dataset(path, text_column, label_column, label_map, drop_unsupported_labels)
    logger.info("Class distribution before training: %s", report.class_counts)
    result = train_and_evaluate(df, report, dataset_name=path.name, seed=seed, test_size=test_size)
    if save:
        save_artifacts(result.pipeline, result.metrics, result.metadata)
    return result

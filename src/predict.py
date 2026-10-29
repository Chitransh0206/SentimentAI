"""Single-text and batch inference on top of the saved Pipeline."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config import settings
from src.data_loader import DataValidationError
from src.preprocessing import is_blank, is_missing


class InvalidInputError(ValueError):
    """User-supplied text is not acceptable (empty, wrong type, too long)."""


@dataclass(frozen=True)
class PredictionResult:
    label: str
    scores: dict[str, float]  # per-class values; meaning given by ``score_kind``
    score_kind: str  # "probability" | "decision_score" | "none"


@dataclass
class BatchSummary:
    total_rows: int = 0
    valid_rows: int = 0
    skipped_rows: int = 0
    skipped_reasons: dict[str, int] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    percentages: dict[str, float] = field(default_factory=dict)


def validate_text(text: Any, max_chars: int | None = None) -> str:
    """Return stripped text or raise ``InvalidInputError`` with a friendly message."""
    max_chars = settings.MAX_TEXT_CHARS if max_chars is None else max_chars
    if is_missing(text) or not isinstance(text, str):
        raise InvalidInputError("Please provide the review as text.")
    stripped = text.strip()
    if not stripped:
        raise InvalidInputError("The review is empty. Please type or paste some text.")
    if len(stripped) > max_chars:
        raise InvalidInputError(f"The review has {len(stripped):,} characters; the limit is {max_chars:,}.")
    return stripped


class SentimentPredictor:
    """Thin wrapper that hides probability/decision-score differences."""

    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline
        self.classes: list[str] = [str(c) for c in pipeline.classes_]
        self.supports_proba = hasattr(pipeline, "predict_proba")
        self.supports_decision = hasattr(pipeline, "decision_function")

    @property
    def score_kind(self) -> str:
        if self.supports_proba:
            return "probability"
        return "decision_score" if self.supports_decision else "none"

    def _score_matrix(self, texts: Sequence[str]) -> np.ndarray | None:
        if self.supports_proba:
            return np.asarray(self.pipeline.predict_proba(texts))
        if self.supports_decision:
            scores = np.asarray(self.pipeline.decision_function(texts))
            return scores.reshape(len(texts), -1)
        return None

    def predict_one(self, text: Any) -> PredictionResult:
        clean = validate_text(text)
        try:
            label = str(self.pipeline.predict([clean])[0])
            matrix = self._score_matrix([clean])
        except Exception as exc:  # model-level failure; keep the UI message friendly
            raise RuntimeError("The model could not score this text.") from exc
        scores = {c: float(v) for c, v in zip(self.classes, matrix[0])} if matrix is not None else {}
        return PredictionResult(label, scores, self.score_kind if scores else "none")

    def predict_texts(self, texts: Sequence[str]) -> tuple[list[str], np.ndarray | None]:
        """Vectorised prediction for already-validated texts."""
        if len(texts) == 0:
            return [], None
        labels = [str(x) for x in self.pipeline.predict(texts)]
        return labels, self._score_matrix(texts)


def _unique_name(name: str, existing: Sequence[str]) -> str:
    candidate = name
    while candidate in existing:
        candidate = f"sentimentai_{candidate}"
    return candidate


def _row_status(value: Any, max_chars: int) -> tuple[str | None, str]:
    """Return (text_to_predict, status). ``text_to_predict`` is None when skipped."""
    if is_blank(value):
        return None, "skipped: empty text"
    text = str(value).strip()
    if len(text) > max_chars:
        return None, f"skipped: longer than {max_chars} characters"
    return text, "ok"


def predict_dataframe(
    predictor: SentimentPredictor,
    df: pd.DataFrame,
    text_column: str,
    chunk_size: int = settings.BATCH_CHUNK_SIZE,
    max_chars: int | None = None,
) -> tuple[pd.DataFrame, BatchSummary]:
    """Predict every valid row; keep ALL original rows/columns and add result columns."""
    max_chars = settings.MAX_TEXT_CHARS if max_chars is None else max_chars
    if text_column not in df.columns:
        raise DataValidationError(f"Column '{text_column}' not found. Available: {list(df.columns)}")
    if df.empty:
        raise DataValidationError("There are no rows to analyze.")

    checked = [_row_status(v, max_chars) for v in df[text_column].tolist()]
    texts = [t for t, _ in checked]
    valid_pos = [i for i, t in enumerate(texts) if t is not None]

    labels_out: list[str | None] = [None] * len(df)
    score_cols = {c: [np.nan] * len(df) for c in predictor.classes}
    has_scores = False
    for start in range(0, len(valid_pos), chunk_size):  # chunking bounds peak memory
        positions = valid_pos[start:start + chunk_size]
        labels, matrix = predictor.predict_texts([texts[i] for i in positions])
        for j, pos in enumerate(positions):
            labels_out[pos] = labels[j]
            if matrix is not None:
                has_scores = True
                for k, cls in enumerate(predictor.classes):
                    score_cols[cls][pos] = float(matrix[j, k])

    result = df.copy()
    existing = list(result.columns)
    label_col = _unique_name("predicted_sentiment", existing)
    status_col = _unique_name("prediction_status", existing + [label_col])
    result[label_col] = labels_out
    result[status_col] = [s for _, s in checked]
    if has_scores:
        prefix = "probability_" if predictor.score_kind == "probability" else "score_"
        for cls in predictor.classes:
            result[_unique_name(prefix + cls, list(result.columns))] = score_cols[cls]

    summary = BatchSummary(total_rows=len(df), valid_rows=len(valid_pos))
    summary.skipped_rows = summary.total_rows - summary.valid_rows
    for _, status in checked:
        if status != "ok":
            summary.skipped_reasons[status] = summary.skipped_reasons.get(status, 0) + 1
    valid_labels = [x for x in labels_out if x is not None]
    summary.counts = {c: valid_labels.count(c) for c in predictor.classes}
    total = max(len(valid_labels), 1)
    summary.percentages = {c: round(100 * n / total, 2) for c, n in summary.counts.items()}
    return result, summary


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    """UTF-8 with BOM so Excel on Windows shows non-ASCII characters correctly."""
    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8-sig")

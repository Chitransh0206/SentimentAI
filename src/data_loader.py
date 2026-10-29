"""Dataset loading, validation and cleaning (training data and CSV uploads)."""
from __future__ import annotations

import io
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Mapping

import pandas as pd

from config import settings
from src.preprocessing import is_blank

logger = logging.getLogger(__name__)

# Spellings accepted for each canonical label (compared case-insensitively).
# Numeric codes such as 0/1/2 are NOT guessed because their meaning differs
# between datasets; pass ``label_map`` explicitly if you need them.
DEFAULT_LABEL_MAP: dict[str, str] = {
    "positive": "positive", "pos": "positive", "good": "positive", "favorable": "positive",
    "negative": "negative", "neg": "negative", "bad": "negative", "unfavorable": "negative",
    "neutral": "neutral", "neu": "neutral",
}


class DataValidationError(ValueError):
    """The dataset or upload is unusable; the message says why and how to fix it."""


class InsufficientDataError(DataValidationError):
    """Not enough (or too unbalanced) data to train a meaningful model."""


@dataclass
class CleaningReport:
    """What happened to the rows while cleaning (never silently drop data)."""

    rows_read: int = 0
    removed_missing_text: int = 0
    removed_unsupported_label: int = 0
    removed_exact_duplicates: int = 0
    removed_conflicting_duplicates: int = 0
    rows_final: int = 0
    unsupported_labels_seen: list[str] = field(default_factory=list)
    class_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def normalize_label(value: object, label_map: Mapping[str, str] | None = None) -> str | None:
    """Map a raw label to positive/negative/neutral, or ``None`` if unsupported."""
    mapping = DEFAULT_LABEL_MAP if label_map is None else {k.strip().lower(): v for k, v in label_map.items()}
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return mapping.get(str(value).strip().lower())


def _read_csv_file(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise DataValidationError(
            f"Dataset not found: {path}\n"
            "Prepare one with `python scripts/prepare_data.py` (see data/README.md) "
            "or pass --data <path-to-csv>."
        )
    if not path.is_file():
        raise DataValidationError(f"Dataset path is not a file: {path}")
    try:
        return pd.read_csv(path, dtype=str, encoding="utf-8-sig", keep_default_na=True)
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError(f"Dataset file is empty: {path}") from exc
    except UnicodeDecodeError as exc:
        raise DataValidationError(f"Dataset must be UTF-8 encoded: {path}") from exc
    except (pd.errors.ParserError, OSError) as exc:
        raise DataValidationError(f"Could not parse {path} as CSV: {exc}") from exc


def clean_labeled_dataframe(
    df: pd.DataFrame,
    text_column: str = settings.DEFAULT_TEXT_COLUMN,
    label_column: str = settings.DEFAULT_LABEL_COLUMN,
    label_map: Mapping[str, str] | None = None,
    drop_unsupported_labels: bool = False,
    drop_duplicates: bool = True,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Validate and clean a labelled frame; returns canonical ``text,sentiment`` columns."""
    missing = [c for c in (text_column, label_column) if c not in df.columns]
    if missing:
        raise DataValidationError(
            f"Missing required column(s): {missing}. Available columns: {list(df.columns)}. "
            "Use --text-column / --label-column to map your column names."
        )
    report = CleaningReport(rows_read=len(df))
    work = pd.DataFrame(
        {"text": df[text_column].astype(object), "sentiment": df[label_column].astype(object)}
    )

    blank = work["text"].map(is_blank)
    report.removed_missing_text = int(blank.sum())
    work = work[~blank].copy()
    work["text"] = work["text"].astype(str).str.strip()

    work["sentiment"] = work["sentiment"].map(lambda v: normalize_label(v, label_map))
    bad = work["sentiment"].isna()
    if bad.any():
        raw = df.loc[work.index[bad], label_column]
        seen = sorted({str(v) for v in raw.tolist()})[:10]
        report.unsupported_labels_seen = seen
        if not drop_unsupported_labels:
            raise DataValidationError(
                f"{int(bad.sum())} row(s) have unsupported labels (e.g. {seen}). Supported labels: "
                f"{sorted(set(DEFAULT_LABEL_MAP))}. Fix the data, supply a label map, or pass "
                "--drop-unsupported to remove those rows explicitly."
            )
        report.removed_unsupported_label = int(bad.sum())
        work = work[~bad].copy()

    if drop_duplicates and len(work):
        # Same text with different labels is ambiguous: drop all copies so it can't
        # appear on both sides of the train/test split with contradicting answers.
        n_labels = work.groupby("text")["sentiment"].transform("nunique")
        conflict = n_labels > 1
        report.removed_conflicting_duplicates = int(conflict.sum())
        work = work[~conflict]
        before = len(work)
        work = work.drop_duplicates(subset=["text", "sentiment"])
        report.removed_exact_duplicates = before - len(work)

    work = work.reset_index(drop=True)
    report.rows_final = len(work)
    report.class_counts = {k: int(v) for k, v in work["sentiment"].value_counts().items()}
    logger.info("Cleaning report: %s", report.to_dict())
    return work, report


def load_dataset(
    path: str | Path,
    text_column: str = settings.DEFAULT_TEXT_COLUMN,
    label_column: str = settings.DEFAULT_LABEL_COLUMN,
    label_map: Mapping[str, str] | None = None,
    drop_unsupported_labels: bool = False,
) -> tuple[pd.DataFrame, CleaningReport]:
    """Read a labelled CSV from disk and clean it."""
    raw = _read_csv_file(Path(path))
    return clean_labeled_dataframe(raw, text_column, label_column, label_map, drop_unsupported_labels)


def ratings_to_sentiment(ratings: pd.Series, negative_max: float = 2, positive_min: float = 4) -> pd.Series:
    """Convert star ratings to labels (<=negative_max negative, >=positive_min positive, else neutral).

    Unparseable ratings become ``None`` so they are reported as unsupported later.
    """
    if negative_max >= positive_min:
        raise DataValidationError("negative_max must be smaller than positive_min.")
    numeric = pd.to_numeric(ratings, errors="coerce")

    def convert(value: float) -> str | None:
        if pd.isna(value):
            return None
        if value <= negative_max:
            return "negative"
        return "positive" if value >= positive_min else "neutral"

    return numeric.map(convert).astype(object)


def class_balance_warnings(class_counts: Mapping[str, int]) -> list[str]:
    """Human-readable warnings about size and class imbalance (does not raise)."""
    warnings: list[str] = []
    total = sum(class_counts.values())
    if total and total < settings.RECOMMENDED_TOTAL_SAMPLES:
        warnings.append(
            f"Only {total} usable rows (< {settings.RECOMMENDED_TOTAL_SAMPLES}); the resulting "
            "metrics are not a credible estimate of real-world performance."
        )
    counts = [class_counts.get(label, 0) for label in settings.LABELS]
    if all(c > 0 for c in counts):
        ratio = max(counts) / min(counts)
        if ratio >= settings.IMBALANCE_RATIO_WARNING:
            warnings.append(
                f"Severe class imbalance: largest/smallest class ratio is {ratio:.1f}:1 "
                f"({dict(class_counts)}). Class weights are used where supported; compare macro F1, not accuracy."
            )
    return warnings


def validate_class_sufficiency(class_counts: Mapping[str, int]) -> None:
    """Raise ``InsufficientDataError`` unless every label has enough examples."""
    problems = []
    for label in settings.LABELS:
        n = int(class_counts.get(label, 0))
        if n < settings.MIN_SAMPLES_PER_CLASS:
            problems.append(f"'{label}' has {n} example(s)")
    if problems:
        raise InsufficientDataError(
            f"Each class needs at least {settings.MIN_SAMPLES_PER_CLASS} examples, but "
            + "; ".join(problems) + ". Add more labelled data."
        )


def read_uploaded_csv(data: bytes, max_mb: int | None = None, max_rows: int | None = None) -> pd.DataFrame:
    """Parse uploaded CSV bytes safely. The content is only ever parsed as data."""
    max_mb = settings.MAX_UPLOAD_MB if max_mb is None else max_mb
    max_rows = settings.MAX_BATCH_ROWS if max_rows is None else max_rows
    if not data or not data.strip():
        raise DataValidationError("The uploaded file is empty.")
    if len(data) > max_mb * 1024 * 1024:
        raise DataValidationError(f"The file is larger than the {max_mb} MB limit.")
    try:
        df = pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8-sig", keep_default_na=True)
    except UnicodeDecodeError as exc:
        raise DataValidationError("The file is not valid UTF-8. Re-save it as 'CSV UTF-8'.") from exc
    except pd.errors.EmptyDataError as exc:
        raise DataValidationError("The file contains no columns or rows.") from exc
    except (pd.errors.ParserError, ValueError) as exc:
        raise DataValidationError(f"The file could not be parsed as CSV: {exc}") from exc
    if df.shape[1] == 0 or df.empty:
        raise DataValidationError("The CSV has a header but no data rows.")
    if len(df) > max_rows:
        raise DataValidationError(f"The file has {len(df):,} rows; the limit is {max_rows:,}.")
    return df

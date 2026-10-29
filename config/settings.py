"""Central configuration for SentimentAI.

All filesystem paths are resolved relative to the project root (the folder that
contains ``app.py``), so the code works no matter which directory it is started from.

A few values can be overridden with environment variables (see ``.env.example``).
Environment variables are read when this module's functions are *called*, so tests
can redirect artifacts to a temporary folder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

BASE_DIR: Path = Path(__file__).resolve().parent.parent

# --- Labels -------------------------------------------------------------------
# Alphabetical on purpose: scikit-learn stores ``classes_`` in sorted order, so
# using the same order everywhere keeps labels, probabilities and confusion
# matrices consistent.
LABELS: tuple[str, ...] = ("negative", "neutral", "positive")

# --- Data ---------------------------------------------------------------------
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
DEFAULT_TEXT_COLUMN = "text"
DEFAULT_LABEL_COLUMN = "sentiment"
DEMO_DATASET_NAME = "demo_reviews.csv"  # filename prefix "demo_" marks demonstration data


def default_dataset_path() -> Path:
    """Training dataset path (override with SENTIMENTAI_DATASET)."""
    override = os.environ.get("SENTIMENTAI_DATASET")
    return Path(override).expanduser() if override else PROCESSED_DATA_DIR / "reviews.csv"


# --- Training -----------------------------------------------------------------
RANDOM_SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
SELECTION_METRIC = "f1_macro"  # mean cross-validated macro F1 on the TRAINING split
MIN_SAMPLES_PER_CLASS = 20  # hard minimum, below this training is refused
RECOMMENDED_TOTAL_SAMPLES = 1000  # below this we warn that results are not credible
IMBALANCE_RATIO_WARNING = 3.0  # largest/smallest class ratio that triggers a warning

# --- Input limits -------------------------------------------------------------
MAX_TEXT_CHARS = int(os.environ.get("SENTIMENTAI_MAX_TEXT_CHARS", "5000"))
MAX_UPLOAD_MB = int(os.environ.get("SENTIMENTAI_MAX_UPLOAD_MB", "25"))
MAX_BATCH_ROWS = int(os.environ.get("SENTIMENTAI_MAX_BATCH_ROWS", "100000"))
BATCH_CHUNK_SIZE = 10_000
MAX_HISTORY_ITEMS = 200


# --- Artifacts ----------------------------------------------------------------
@dataclass(frozen=True)
class ArtifactPaths:
    """Where the trained model and its reports live."""

    model: Path
    metrics: Path
    metadata: Path

    @property
    def directories(self) -> tuple[Path, ...]:
        return tuple({self.model.parent, self.metrics.parent, self.metadata.parent})


def get_artifact_paths() -> ArtifactPaths:
    """Resolve artifact locations (override root with SENTIMENTAI_ARTIFACT_ROOT)."""
    override = os.environ.get("SENTIMENTAI_ARTIFACT_ROOT")
    root = Path(override).expanduser().resolve() if override else BASE_DIR
    return ArtifactPaths(
        model=root / "models" / "sentiment_pipeline.joblib",
        metrics=root / "artifacts" / "metrics.json",
        metadata=root / "artifacts" / "model_metadata.json",
    )

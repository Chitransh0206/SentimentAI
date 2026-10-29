"""Saving and loading model artifacts.

SECURITY: ``joblib.load`` unpickles data and can execute arbitrary code. Only load
artifacts that YOU created with ``scripts/train_model.py``. This application never
loads a model uploaded through the UI.
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from config.settings import ArtifactPaths, get_artifact_paths

logger = logging.getLogger(__name__)

TRAIN_HINT = "Train a model first:  python scripts/train_model.py   (see README.md, section 'Train the model')."


class ModelNotFoundError(FileNotFoundError):
    """The trained pipeline (or a report) does not exist yet."""


def _atomic_write(path: Path, writer) -> None:
    """Write via a temp file + rename so a crash never leaves a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        writer(Path(tmp_name))
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.remove(tmp_name)


def save_artifacts(pipeline: Pipeline, metrics: dict, metadata: dict, paths: ArtifactPaths | None = None) -> ArtifactPaths:
    """Persist the full pipeline plus JSON reports."""
    paths = paths or get_artifact_paths()
    _atomic_write(paths.model, lambda p: joblib.dump(pipeline, p, compress=3))
    _atomic_write(paths.metrics, lambda p: p.write_text(json.dumps(metrics, indent=2), encoding="utf-8"))
    _atomic_write(paths.metadata, lambda p: p.write_text(json.dumps(metadata, indent=2), encoding="utf-8"))
    logger.info("Saved artifacts to %s", paths)
    return paths


def load_pipeline(path: Path | None = None) -> Pipeline:
    """Load the trusted, locally trained pipeline."""
    path = Path(path) if path else get_artifact_paths().model
    if not path.is_file():
        raise ModelNotFoundError(f"No trained model found at {path}. {TRAIN_HINT}")
    try:
        pipeline = joblib.load(path)
    except Exception as exc:  # corrupt file / incompatible scikit-learn version
        raise ModelNotFoundError(
            f"The model file at {path} could not be loaded ({type(exc).__name__}). "
            f"It may be corrupt or built with a different scikit-learn version. {TRAIN_HINT}"
        ) from exc
    if not isinstance(pipeline, Pipeline) or not hasattr(pipeline, "classes_"):
        raise ModelNotFoundError(f"{path} does not contain a fitted scikit-learn Pipeline. {TRAIN_HINT}")
    return pipeline


def load_json_artifact(path: Path) -> dict[str, Any] | None:
    """Return a JSON report, or ``None`` when it is missing/unreadable."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None

"""Streamlit glue: cached model loading, session history, shared page setup.

Pure helpers (history, counts) take the session dict as an argument so they can be
unit-tested without a running Streamlit server.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import MutableMapping

import pandas as pd
import streamlit as st

from config import settings
from src.model_utils import TRAIN_HINT, ModelNotFoundError, load_json_artifact, load_pipeline
from src.predict import BatchSummary, PredictionResult, SentimentPredictor

HISTORY_KEY = "history"
BATCH_RUNS_KEY = "batch_runs"

_CSS = """
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
h1, h2, h3 {letter-spacing: -0.01em;}
.sa-result {border-radius: 12px; padding: 1.1rem 1.3rem; border-left: 8px solid var(--c); background: rgba(128,128,128,.08);}
.sa-result h2 {margin: 0; color: var(--c);}
.sa-muted {color: #8A94A6; font-size: .9rem;}
</style>
"""


def setup_page(title: str) -> None:
    """Must be the first Streamlit call on every page."""
    st.set_page_config(page_title=f"{title} · SentimentAI", page_icon="📊", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading model...")
def _load_predictor_cached(model_path: str, mtime_ns: int) -> SentimentPredictor:
    # ``mtime_ns`` is part of the cache key so retraining invalidates the cache.
    return SentimentPredictor(load_pipeline(Path(model_path)))


def get_predictor() -> SentimentPredictor | None:
    """Return the cached predictor, or ``None`` if no model has been trained."""
    path = settings.get_artifact_paths().model
    if not path.is_file():
        return None
    try:
        return _load_predictor_cached(str(path), path.stat().st_mtime_ns)
    except ModelNotFoundError as exc:
        st.error(str(exc))
        return None


def get_reports() -> tuple[dict | None, dict | None]:
    """(metrics, metadata) read from the generated JSON artifacts."""
    paths = settings.get_artifact_paths()
    return load_json_artifact(paths.metrics), load_json_artifact(paths.metadata)


def require_predictor() -> SentimentPredictor:
    """Return the predictor or show training instructions and stop the page."""
    predictor = get_predictor()
    if predictor is None:
        st.warning("No trained model was found, so predictions are unavailable.")
        st.markdown("Train a model first (PowerShell, from the project folder):")
        st.code("python scripts/prepare_data.py --demo\npython scripts/train_model.py --data data/raw/demo_reviews.csv", language="powershell")
        st.caption(TRAIN_HINT)
        st.stop()
    return predictor


# --- session state helpers (per browser session, never shared) -----------------
def init_session(state: MutableMapping) -> None:
    state.setdefault(HISTORY_KEY, [])
    state.setdefault(BATCH_RUNS_KEY, [])


def add_history(state: MutableMapping, text: str, result: PredictionResult) -> None:
    init_session(state)
    top = max(result.scores.values()) if result.scores and result.score_kind == "probability" else None
    state[HISTORY_KEY].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"),
        "review": text if len(text) <= 120 else text[:117] + "...",
        "sentiment": result.label,
        "top_probability": round(top, 3) if top is not None else None,
    })
    del state[HISTORY_KEY][settings.MAX_HISTORY_ITEMS:]


def add_batch_run(state: MutableMapping, filename: str, summary: BatchSummary) -> None:
    init_session(state)
    state[BATCH_RUNS_KEY].insert(0, {
        "time": datetime.now().strftime("%H:%M:%S"), "file": filename,
        "valid": summary.valid_rows, "skipped": summary.skipped_rows, "counts": dict(summary.counts),
    })


def session_label_counts(state: MutableMapping) -> dict[str, int]:
    """Totals across single predictions and batch runs of this session."""
    init_session(state)
    counts = {label: 0 for label in settings.LABELS}
    for item in state[HISTORY_KEY]:
        counts[item["sentiment"]] = counts.get(item["sentiment"], 0) + 1
    for run in state[BATCH_RUNS_KEY]:
        for label, n in run["counts"].items():
            counts[label] = counts.get(label, 0) + n
    return counts


def history_frame(state: MutableMapping) -> pd.DataFrame:
    init_session(state)
    return pd.DataFrame(state[HISTORY_KEY], columns=["time", "review", "sentiment", "top_probability"])


def guess_text_column(columns: list[str]) -> int:
    """Index of the most likely review column (falls back to 0)."""
    preferred = ("text", "review", "reviews", "comment", "content", "body", "sentence")
    lowered = [c.strip().lower() for c in columns]
    for name in preferred:
        if name in lowered:
            return lowered.index(name)
    return 0

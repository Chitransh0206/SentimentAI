import os

import pytest

from src.app_helpers import (
    add_batch_run, add_history, guess_text_column, history_frame, session_label_counts,
)
from src.predict import BatchSummary, PredictionResult
from src.visualization import (
    confusion_matrix_fig, model_comparison_fig, per_class_fig, score_bar_fig, sentiment_distribution_fig,
)


def test_history_and_counts_are_per_state_dict():
    a, b = {}, {}
    add_history(a, "x" * 300, PredictionResult("positive", {"positive": 0.9, "negative": 0.05, "neutral": 0.05}, "probability"))
    add_batch_run(a, "f.csv", BatchSummary(valid_rows=3, counts={"negative": 2, "positive": 1}))
    counts = session_label_counts(a)
    assert counts == {"negative": 2, "neutral": 0, "positive": 2}
    assert session_label_counts(b) == {"negative": 0, "neutral": 0, "positive": 0}  # sessions are isolated
    frame = history_frame(a)
    assert len(frame) == 1 and len(frame.loc[0, "review"]) <= 120


def test_history_is_capped():
    state = {}
    for _ in range(250):
        add_history(state, "t", PredictionResult("neutral", {}, "none"))
    assert len(state["history"]) == 200


def test_guess_text_column():
    assert guess_text_column(["id", "Review", "stars"]) == 1
    assert guess_text_column(["a", "b"]) == 0


def test_charts_handle_empty_data():
    assert len(sentiment_distribution_fig({}).layout.annotations) == 1
    assert len(score_bar_fig({}, "none").layout.annotations) == 1
    assert len(confusion_matrix_fig(None, []).layout.annotations) == 1
    assert len(model_comparison_fig(None).layout.annotations) == 1
    assert len(per_class_fig(None).layout.annotations) == 1


def test_charts_with_data():
    assert sentiment_distribution_fig({"positive": 2, "negative": 1}).data
    assert score_bar_fig({"positive": 0.7, "negative": 0.2, "neutral": 0.1}, "probability").data
    assert confusion_matrix_fig([[1, 0], [0, 1]], ["a", "b"]).data


# --- Streamlit smoke tests (run the real scripts headlessly) -------------------
from pathlib import Path  # noqa: E402

from streamlit.testing.v1 import AppTest  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def run_page(page: str) -> AppTest:
    # AppTest resolves relative paths against this file, so use absolute paths.
    return AppTest.from_file(str(ROOT / page), default_timeout=30).run()

PAGES = ["app.py", "pages/1_Analyze_Text.py", "pages/2_Batch_Analysis.py", "pages/3_Model_Performance.py", "pages/4_About.py"]


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_without_model(page, artifact_root):
    at = run_page(page)
    assert not at.exception
    if page != "app.py" and page != "pages/4_About.py":
        assert any("model" in w.value.lower() for w in at.warning)


def test_single_analysis_flow_with_model(training_result, artifact_root):
    from src.model_utils import save_artifacts
    save_artifacts(training_result.pipeline, training_result.metrics, training_result.metadata)
    at = run_page("pages/1_Analyze_Text.py")
    at.text_area[0].set_value("The product is excellent.").run()
    at.button[0].click().run()
    assert not at.exception
    assert len(at.dataframe) == 1  # history table shows the analysed review
    # empty input -> friendly warning, no crash
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()
    assert not at.exception and any("empty" in w.value.lower() for w in at.warning)
    # other pages render with artifacts present
    for page in ("app.py", "pages/3_Model_Performance.py", "pages/4_About.py"):
        assert not run_page(page).exception

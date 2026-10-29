"""Analyze a single review."""
import streamlit as st

from config import settings
from src.app_helpers import add_history, history_frame, init_session, require_predictor, setup_page
from src.predict import InvalidInputError
from src.visualization import COLORS, score_bar_fig

setup_page("Analyze Text")
init_session(st.session_state)
st.title("Analyze Text")
predictor = require_predictor()

EXPLANATIONS = {
    "positive": "The wording is mostly favourable (praise, satisfaction, recommendations).",
    "negative": "The wording is mostly unfavourable (complaints, disappointment, negations of praise).",
    "neutral": "The text is factual, mixed, or lacks strong emotional wording.",
}

review = st.text_area("Review text", height=160, max_chars=settings.MAX_TEXT_CHARS,
                      placeholder="e.g. The battery life is not good and support never replied.",
                      key="single_text")
if st.button("Analyze Sentiment", type="primary"):
    try:
        result = predictor.predict_one(review)
    except InvalidInputError as exc:
        st.warning(str(exc))
    except RuntimeError:
        st.error("Something went wrong while analyzing this text. Please try different text.")
    else:
        add_history(st.session_state, review.strip(), result)
        color = COLORS.get(result.label, "#4F6BED")
        st.markdown(
            f'<div class="sa-result" style="--c:{color}"><span class="sa-muted">Predicted sentiment</span>'
            f"<h2>{result.label.title()}</h2></div>", unsafe_allow_html=True)
        st.write(EXPLANATIONS.get(result.label, ""))
        if result.scores:
            st.plotly_chart(score_bar_fig(result.scores, result.score_kind), width="stretch")
            if result.score_kind == "probability":
                st.caption("Values are probabilities estimated by the model. They are not a guarantee that the prediction is correct.")
            else:
                st.caption("Raw decision scores, not probabilities. Higher means the class is favoured.")

st.subheader("Session history")
history = history_frame(st.session_state)
if history.empty:
    st.info("Analyzed reviews appear here. History lives only in this browser session.")
else:
    st.dataframe(history, hide_index=True, width="stretch")
    if st.button("Clear history"):
        st.session_state["history"] = []
        st.rerun()

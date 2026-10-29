"""SentimentAI - dashboard (home page). Run with: python -m streamlit run app.py"""
import streamlit as st

from src.app_helpers import get_predictor, get_reports, init_session, session_label_counts, setup_page
from src.visualization import sentiment_distribution_fig

setup_page("Dashboard")
init_session(st.session_state)

st.title("SentimentAI")
st.caption("Classify review text as Positive, Negative or Neutral with a TF-IDF + scikit-learn model. Runs fully offline.")

predictor = get_predictor()
metrics, metadata = get_reports()

with st.container(border=True):
    st.subheader("Model status")
    if predictor is None:
        st.warning("No trained model found. Train one to enable the analysis pages.")
        st.code("python scripts/prepare_data.py --demo\npython scripts/train_model.py --data data/raw/demo_reviews.csv", language="powershell")
    else:
        st.success("Model loaded and ready.")
        if metadata:
            if metadata.get("is_demonstration_data"):
                st.warning("This model was trained on the synthetic demo dataset. It demonstrates the software only; "
                           "its metrics are not a real-world benchmark.")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Selected model", metadata.get("model_name", "-"))
            c2.metric("Training rows", f"{metadata.get('n_train', 0):,}")
            c3.metric("Test rows", f"{metadata.get('n_test', 0):,}")
            test = (metrics or {}).get("test", {})
            c4.metric("Test macro F1", f"{test['f1_macro']:.3f}" if "f1_macro" in test else "-")
            st.caption(f"Dataset: {metadata.get('dataset_name')} · trained {metadata.get('trained_at_utc')} UTC")

st.subheader("This session")
counts = session_label_counts(st.session_state)
total = sum(counts.values())
left, right = st.columns([1, 1])
with left:
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Analyzed", total)
    m2.metric("Positive", counts["positive"])
    m3.metric("Neutral", counts["neutral"])
    m4.metric("Negative", counts["negative"])
    st.caption("Session statistics are kept in memory for this browser tab only and are lost on refresh.")
with right:
    st.plotly_chart(sentiment_distribution_fig(counts, "Session sentiment distribution"), width="stretch")

st.subheader("Tools")
a, b, c, d = st.columns(4)
a.page_link("pages/1_Analyze_Text.py", label="Analyze Text", icon="📝")
b.page_link("pages/2_Batch_Analysis.py", label="Batch Analysis", icon="📂")
c.page_link("pages/3_Model_Performance.py", label="Model Performance", icon="📈")
d.page_link("pages/4_About.py", label="About", icon="ℹ️")

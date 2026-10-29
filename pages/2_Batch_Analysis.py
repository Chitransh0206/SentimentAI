"""Analyze a CSV of reviews."""
import streamlit as st

from config import settings
from src.app_helpers import add_batch_run, guess_text_column, init_session, require_predictor, setup_page
from src.data_loader import DataValidationError, read_uploaded_csv
from src.predict import predict_dataframe, to_csv_bytes
from src.visualization import sentiment_distribution_fig

setup_page("Batch Analysis")
init_session(st.session_state)
st.title("Batch Analysis")
predictor = require_predictor()

st.write(f"Upload a UTF-8 CSV (max {settings.MAX_UPLOAD_MB} MB, {settings.MAX_BATCH_ROWS:,} rows) and choose the column with the review text.")
upload = st.file_uploader("CSV file", type=["csv"])

if upload is None:
    st.info("No file selected yet.")
    st.session_state.pop("batch_result", None)
    st.stop()

try:
    df = read_uploaded_csv(upload.getvalue())
except DataValidationError as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Preview")
st.dataframe(df.head(20), width="stretch")
st.caption(f"{len(df):,} rows · {df.shape[1]} columns")
column = st.selectbox("Text column", list(df.columns), index=guess_text_column(list(df.columns)))

if st.button("Analyze Batch", type="primary"):
    try:
        with st.spinner("Analyzing..."):
            result_df, summary = predict_dataframe(predictor, df, column)
    except DataValidationError as exc:
        st.error(str(exc))
    else:
        add_batch_run(st.session_state, upload.name, summary)
        # Kept in session_state because the download button triggers a rerun.
        st.session_state["batch_result"] = (upload.name, column, result_df, summary)

stored = st.session_state.get("batch_result")
if stored and stored[0] == upload.name and stored[1] == column:
    _, _, result_df, summary = stored
    st.subheader("Summary")
    m = st.columns(4)
    m[0].metric("Total records", f"{summary.total_rows:,}")
    m[1].metric("Valid records", f"{summary.valid_rows:,}")
    m[2].metric("Empty / invalid", f"{summary.skipped_rows:,}")
    m[3].metric("Distinct skip reasons", len(summary.skipped_reasons))
    if summary.skipped_rows:
        st.warning("Rows below were kept in the output but not scored: "
                   + "; ".join(f"{n} × {reason}" for reason, n in summary.skipped_reasons.items()))
    p = st.columns(3)
    for col, label in zip(p, settings.LABELS):
        col.metric(label.title(), f"{summary.counts.get(label, 0):,}", f"{summary.percentages.get(label, 0):.1f}%", delta_color="off")
    st.plotly_chart(sentiment_distribution_fig(summary.counts, "Batch sentiment distribution"), width="stretch")
    st.dataframe(result_df.head(100), width="stretch")
    st.download_button("Download predictions (CSV)", to_csv_bytes(result_df),
                       file_name="sentimentai_predictions.csv", mime="text/csv")

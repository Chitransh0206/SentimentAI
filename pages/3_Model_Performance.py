"""Evaluation results read from the generated artifacts."""
import pandas as pd
import streamlit as st

from src.app_helpers import get_reports, setup_page
from src.visualization import confusion_matrix_fig, model_comparison_fig, per_class_fig

setup_page("Model Performance")
st.title("Model Performance")
metrics, metadata = get_reports()

if not metrics or not metadata:
    st.warning("No evaluation results found. Train a model to generate them.")
    st.code("python scripts/prepare_data.py --demo\npython scripts/train_model.py --data data/raw/demo_reviews.csv", language="powershell")
    st.stop()

if metadata.get("is_demonstration_data"):
    st.error("Trained on synthetic demonstration data. These numbers only show that the pipeline works; "
             "they are NOT a real-world performance estimate.")
for w in metadata.get("warnings", []):
    st.warning(w)

test = metrics["test"]
st.subheader("Held-out test results")
st.caption(f"Selected model: **{metrics['selected_model']}**. Test set: {test['n_samples']:,} rows never used for fitting or model selection.")
cols = st.columns(5)
for col, (name, key) in zip(cols, [("Accuracy", "accuracy"), ("Macro precision", "precision_macro"),
                                  ("Macro recall", "recall_macro"), ("Macro F1", "f1_macro"), ("Weighted F1", "f1_weighted")]):
    col.metric(name, f"{test[key]:.3f}")

tab1, tab2, tab3, tab4 = st.tabs(["Per class", "Confusion matrix", "Model comparison (validation)", "Training metadata"])
with tab1:
    st.dataframe(pd.DataFrame(test["per_class"]).T.rename_axis("class"), width="stretch")
    st.plotly_chart(per_class_fig(test["per_class"]), width="stretch")
    with st.expander("Full classification report"):
        st.code(test["classification_report_text"])
with tab2:
    st.plotly_chart(confusion_matrix_fig(test["confusion_matrix"], test["labels"]), width="stretch")
    st.caption("Rows are the true class, columns the predicted class. The diagonal shows correct predictions.")
with tab3:
    st.caption(f"Validation = {metrics['selection_metric']}. These scores chose the model; they are separate from the test results above.")
    st.plotly_chart(model_comparison_fig(metrics["validation"]), width="stretch")
    st.dataframe(pd.DataFrame(metrics["validation"]), hide_index=True, width="stretch")
with tab4:
    skip = {"cleaning_report", "warnings", "versions"}
    st.json({k: v for k, v in metadata.items() if k not in skip})
    st.write("Cleaning report"); st.json(metadata.get("cleaning_report") or {})
    st.write("Library versions"); st.json(metadata.get("versions") or {})

with st.expander("What do these metrics mean?"):
    st.markdown("""
- **Accuracy**: share of all reviews classified correctly. Misleading when classes are imbalanced.
- **Precision**: of the reviews predicted as a class, how many truly belong to it.
- **Recall**: of the reviews that truly belong to a class, how many were found.
- **F1**: harmonic mean of precision and recall. **Macro F1** averages classes equally (used for model selection); **weighted F1** weights by class size.
- **Validation vs test**: validation (cross-validation on the training split) picks the model; the test set is used once at the end.
""")

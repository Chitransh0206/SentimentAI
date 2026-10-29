"""Project documentation inside the app."""
import streamlit as st

from src.app_helpers import get_reports, setup_page

setup_page("About")
st.title("About SentimentAI")
_, metadata = get_reports()

st.markdown("""
### How it works
1. **Preprocess**: Unicode normalisation, lower-casing, URL/HTML removal, whitespace cleanup. Negations such as *not*, *no*, *never* are kept.
2. **TF-IDF**: each review becomes a sparse vector. Every word or two-word phrase (e.g. *not good*) is weighted higher when it is frequent in the review and rare across all reviews.
3. **Classifier**: a linear or probabilistic model maps the vector to Positive, Negative or Neutral. Three candidates are compared by cross-validated macro F1 on the training split; the best is evaluated once on a held-out test set.
4. **Pipeline**: vectorizer and classifier are saved together, so training and inference use identical features.

### Architecture
`data_loader` → `train` (+ `evaluate`) → `model_utils` (Joblib artifacts) → `predict` → Streamlit pages (`app.py`, `pages/`).

### Technologies
Python, scikit-learn, pandas, NumPy, Streamlit, Plotly, Joblib, Pytest.

### Limitations
- Bag-of-words models do not understand sarcasm, context, or long-range dependencies.
- Quality depends entirely on the training data; domain shift (e.g. movie reviews → electronics) hurts accuracy.
- Neutral is the hardest class because it overlaps with mild opinions.
- English text only unless trained on another language.
- Probabilities are estimates and not guarantees.

### Privacy
All processing happens locally. No text is sent to external services, and nothing is stored on disk by the app.

### Security note
Model files are Joblib/pickle artifacts. Only load files you trained yourself; this app never loads uploaded models.

### Reproduce training
```
python scripts/prepare_data.py --demo
python scripts/train_model.py --data data/raw/demo_reviews.csv
```
""")
if metadata:
    st.subheader("Current model")
    st.json({k: metadata[k] for k in ("model_name", "dataset_name", "trained_at_utc", "random_seed", "n_train", "n_test") if k in metadata})

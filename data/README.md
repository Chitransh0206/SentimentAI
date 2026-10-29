# Data

SentimentAI trains on a labelled CSV. **No real dataset is bundled**, and nothing is downloaded at run time.

## Canonical schema (`data/processed/reviews.csv`)

| column | meaning |
|---|---|
| `text` | the review (UTF-8) |
| `sentiment` | `positive`, `negative` or `neutral` (case-insensitive; `pos`/`neg`/`neu` also accepted) |

Numeric labels (0/1/2) are never guessed, because their meaning differs per dataset. Convert them first.

## Options

1. **Demo data (software demo only)**
   `python scripts/prepare_data.py --demo` writes template-generated sentences to `data/raw/demo_reviews.csv`.
   Models trained on it score near 100%, which says nothing about real-world accuracy.
2. **Your own dataset with star ratings** (for example 1-5 stars):
   `python scripts/prepare_data.py --input data\raw\my.csv --text-column reviewText --rating-column overall`
   Default mapping: rating <= 2 negative, 3 neutral, >= 4 positive (`--negative-max`, `--positive-min`).
3. **Your own dataset with sentiment labels:**
   `python scripts/prepare_data.py --input data\raw\my.csv --text-column review --label-column label`

Use any public review dataset whose licence permits your use (for example Amazon or Yelp review dumps). Check the licence and terms yourself; this project does not redistribute data and I could not verify any download source from the build environment.

## Cleaning that is applied (and reported)
Missing/empty text removed; unsupported labels rejected (or dropped with `--drop-unsupported`); exact duplicates removed; texts that appear with conflicting labels removed. Counts for each step are printed and stored in `artifacts/model_metadata.json`.

## Minimum requirements
At least 20 examples per class are required to train; at least ~1,000 rows total are recommended for results to be meaningful. Severe class imbalance (>= 3:1) triggers a warning.

"""Prepare a canonical ``text,sentiment`` CSV.

Examples (PowerShell, from the project folder):

  # 1) Tiny synthetic demo set (software demo only - NOT a benchmark)
  python scripts/prepare_data.py --demo

  # 2) Convert your own dataset that has star ratings (1-5)
  python scripts/prepare_data.py --input data/raw/my_reviews.csv --text-column reviewText --rating-column overall

  # 3) Convert your own dataset that already has sentiment labels
  python scripts/prepare_data.py --input data/raw/my_reviews.csv --text-column review --label-column label
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # makes `config` and `src` importable from any working directory

from config import settings  # noqa: E402
from src.data_loader import (  # noqa: E402
    DataValidationError,
    _read_csv_file,
    class_balance_warnings,
    clean_labeled_dataframe,
    ratings_to_sentiment,
)
from src.demo_data import make_synthetic_reviews  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--demo", action="store_true", help="write the synthetic demo dataset to data/raw/")
    p.add_argument("--input", type=Path, help="source CSV (UTF-8)")
    p.add_argument("--output", type=Path, default=settings.PROCESSED_DATA_DIR / "reviews.csv")
    p.add_argument("--text-column", default=settings.DEFAULT_TEXT_COLUMN)
    p.add_argument("--label-column", help="column that already holds sentiment labels")
    p.add_argument("--rating-column", help="numeric star-rating column to convert to labels")
    p.add_argument("--negative-max", type=float, default=2, help="ratings <= this are negative (default 2)")
    p.add_argument("--positive-min", type=float, default=4, help="ratings >= this are positive (default 4)")
    p.add_argument("--drop-unsupported", action="store_true", help="drop rows with unknown labels (reported)")
    args = p.parse_args()

    try:
        if args.demo:
            out = settings.RAW_DATA_DIR / settings.DEMO_DATASET_NAME
            out.parent.mkdir(parents=True, exist_ok=True)
            make_synthetic_reviews().to_csv(out, index=False, encoding="utf-8")
            print(f"Wrote SYNTHETIC demo data to {out}")
            print("WARNING: template-generated text. Use it only to try the app; metrics are not meaningful.")
            print(f"Next: python scripts/train_model.py --data {out.relative_to(ROOT)}")
            return 0

        if not args.input:
            p.error("provide --demo or --input <csv>")
        if bool(args.label_column) == bool(args.rating_column):
            p.error("provide exactly one of --label-column or --rating-column")

        raw = _read_csv_file(args.input)
        label_column = args.label_column
        if args.rating_column:
            if args.rating_column not in raw.columns:
                raise DataValidationError(f"Rating column '{args.rating_column}' not found. Columns: {list(raw.columns)}")
            raw["__sentiment__"] = ratings_to_sentiment(raw[args.rating_column], args.negative_max, args.positive_min)
            label_column = "__sentiment__"
            args.drop_unsupported = True  # unparseable ratings are dropped and reported

        df, report = clean_labeled_dataframe(raw, args.text_column, label_column, drop_unsupported_labels=args.drop_unsupported)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False, encoding="utf-8")
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print(f"Wrote {len(df):,} rows to {args.output}")
    for key, value in report.to_dict().items():
        print(f"  {key}: {value}")
    for w in class_balance_warnings(report.class_counts):
        print(f"WARNING: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

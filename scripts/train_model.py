"""Train, select, evaluate and save the SentimentAI model.

  python scripts/train_model.py --data data/processed/reviews.csv
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))  # makes `config` and `src` importable from any working directory

from config import settings  # noqa: E402
from src.data_loader import DataValidationError  # noqa: E402
from src.train import run_training  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=settings.default_dataset_path(), help="labelled CSV")
    p.add_argument("--text-column", default=settings.DEFAULT_TEXT_COLUMN)
    p.add_argument("--label-column", default=settings.DEFAULT_LABEL_COLUMN)
    p.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    p.add_argument("--test-size", type=float, default=settings.TEST_SIZE)
    p.add_argument("--drop-unsupported", action="store_true", help="drop rows with unknown labels (reported)")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if not 0.05 <= args.test_size <= 0.5:
        p.error("--test-size must be between 0.05 and 0.5")
    try:
        result = run_training(args.data, args.text_column, args.label_column, None,
                              args.drop_unsupported, args.seed, args.test_size)
    except DataValidationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    paths = settings.get_artifact_paths()
    t = result.metrics["test"]
    print("\n=== Validation (cross-validation on training split) ===")
    for r in result.metrics["validation"]:
        print(f"  {r['model']:<28} macro F1 {r['cv_f1_macro_mean']:.4f} ± {r['cv_f1_macro_std']:.4f}")
    print(f"Selected: {result.metrics['selected_model']}")
    print("\n=== Held-out test set ===")
    print(f"  accuracy {t['accuracy']:.4f} | macro F1 {t['f1_macro']:.4f} | weighted F1 {t['f1_weighted']:.4f}")
    print(t["classification_report_text"])
    for w in result.metadata["warnings"]:
        print(f"WARNING: {w}")
    print(f"Saved model:    {paths.model}\nSaved metrics:  {paths.metrics}\nSaved metadata: {paths.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

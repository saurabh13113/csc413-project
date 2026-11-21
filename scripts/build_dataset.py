"""Build data/processed/: dropped.csv, vocab.json, splits.json, ranges.json, encoded.json.

Usage: python scripts/build_dataset.py [--out DIR]
"""
import argparse
from collections import Counter

from chorales.dataset import build_processed
from chorales.load import PROCESSED_DIR


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=PROCESSED_DIR)
    args = ap.parse_args()
    report, encoded, splits = build_processed(args.out)
    print(f"iterator entries {report.n_entries}, unique {report.n_unique}, "
          f"passed checks {report.n_passed_checks}, kept {len(report.kept)}")
    print("dropped:", dict(Counter(reason.split("=")[0].split(" ")[0] for _, reason in report.dropped)))
    print("splits:", {k: len(v) for k, v in splits.items() if k != "seed"}, "seed", splits["seed"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

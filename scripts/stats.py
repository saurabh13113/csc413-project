"""Dataset statistics (report Table 1 and Figure 1) on the cleaned 327-chorale set.

Usage: python scripts/stats.py [--out DIR]
Writes DIR/stats.json and DIR/chorale_distributions.png (default DIR = data/processed).
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from chorales.augment import VOICE_NAMES, allowed_shifts, compute_ranges, voice_pitches
from chorales.encode import encode
from chorales.load import PROCESSED_DIR, load_chorales
from chorales.split import load_splits
from chorales.vocab import is_pitch

VOICE_LABELS = {"S": "Soprano", "A": "Alto", "T": "Tenor", "B": "Bass"}
COLORS = {"S": "tab:blue", "A": "tab:orange", "T": "tab:green", "B": "tab:red"}


def compute_stats(chorales, splits):
    encoded = {cid: encode(s)["ids"] for cid, s in chorales}
    steps = [(len(ids) - 2) // 4 for ids in encoded.values()]
    tokens = [len(ids) for ids in encoded.values()]
    written_notes = sum(1 for _, s in chorales for p in s.parts for n in p.flatten().notes
                        if n.quarterLength > 0)
    sounding_notes = sum(is_pitch(t) for ids in encoded.values() for t in ids)
    modes = [s.analyze("key").mode for _, s in chorales]
    train_ranges = compute_ranges(encoded[c] for c in splits["train"])
    full_ranges = compute_ranges(encoded.values())
    shifts_train = [len(allowed_shifts(encoded[c], train_ranges)) for c in splits["train"]]
    return {
        "n_chorales": len(chorales),
        "split_sizes": {k: len(splits[k]) for k in ("train", "val", "test")},
        "total_notes_written": written_notes,
        "total_notes_sounding": sounding_notes,
        "length_steps": {"mean": round(mean(steps), 2), "median": float(np.median(steps)),
                         "min": min(steps), "max": max(steps)},
        "token_length": {"mean": round(mean(tokens), 2), "max": max(tokens),
                         "n_over_1025": sum(t > 1025 for t in tokens)},
        "pitch_range_all": full_ranges,
        "pitch_range_train": train_ranges,
        "key_mode": {"major": modes.count("major"), "minor": modes.count("minor"),
                     "major_share": round(modes.count("major") / len(modes), 3)},
        "vocab_size": 51,
        "augmentation": {
            "mean_allowed_shifts_train": round(mean(shifts_train), 3),
            "mean_allowed_shifts_all_full_ranges": round(
                mean(len(allowed_shifts(x, full_ranges)) for x in encoded.values()), 3),
            "train_examples_all_mode": sum(shifts_train),
        },
    }, encoded, steps


def plot(encoded, steps, path):
    pitches = {v: [] for v in VOICE_NAMES}
    for ids in encoded.values():
        for v, ps in zip(VOICE_NAMES, voice_pitches(ids)):
            pitches[v].extend(ps)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.hist(steps, bins=30, color="steelblue", edgecolor="white")
    ax1.set_xlabel("Chorale length (sixteenth-note steps)")
    ax1.set_ylabel("Number of chorales")
    ax1.set_title(f"Distribution of Chorale Lengths (n={len(steps)})")
    ax1.grid(axis="y", alpha=0.25)

    allp = [p for ps in pitches.values() for p in ps]
    bins = np.arange(min(allp) - 0.5, max(allp) + 1.5, 1)
    bottom = np.zeros(len(bins) - 1)
    for v, ps in pitches.items():
        counts, _ = np.histogram(ps, bins=bins)
        ax2.bar(bins[:-1], counts, width=1, bottom=bottom, color=COLORS[v], label=VOICE_LABELS[v],
                align="edge", edgecolor="white", linewidth=0.3)
        bottom += counts
    ax2.set_xlabel("MIDI pitch")
    ax2.set_ylabel("Number of notes (note onsets)")
    ax2.set_title("Pitch Distribution by Voice")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=PROCESSED_DIR, type=Path)
    args = ap.parse_args()
    chorales = load_chorales(write_csv=False)
    splits = load_splits(PROCESSED_DIR / "splits.json")
    stats, encoded, steps = compute_stats(chorales, splits)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    plot(encoded, steps, args.out / "chorale_distributions.png")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()

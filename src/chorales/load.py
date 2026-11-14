"""Load, deduplicate and clean the Bach chorales shipped with music21.

Steps (see plan.md, Phase 1):
1. iterate the chorale filenames, keeping the first occurrence of each
2. drop chorales that don't have exactly 4 parts
3. drop chorales with a note/rest that doesn't fit the 16th-note grid
4. assert no chords inside parts
5. assert mean pitch strictly decreases from part 0 to part 3 (voice v = part index v)
6. drop exact content duplicates
7. write data/processed/dropped.csv
"""
import csv
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

from music21 import chord, corpus, stream

REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
N_VOICES = 4


@dataclass
class LoadReport:
    n_entries: int = 0
    n_unique: int = 0
    n_passed_checks: int = 0
    kept: list = field(default_factory=list)      # [(chorale_id, score)]
    dropped: list = field(default_factory=list)   # [(filename, reason)]


def chorale_id(filename):
    """'bach/bwv269' -> 'bwv269'."""
    return filename.split("/")[-1]


def on_grid(x):
    """True if x quarter notes is a whole number of 16th notes."""
    return (Fraction(x) * 4).denominator == 1


def off_grid_element(score):
    for part in score.parts:
        for el in part.flatten().notesAndRests:
            if not (on_grid(el.offset) and on_grid(el.quarterLength)):
                return el
    return None


def content_key(score):
    """Exact content: (offset, quarterLength, midi or -1) for every note/rest in every part."""
    return tuple(
        tuple(
            (Fraction(el.offset), Fraction(el.quarterLength), el.pitch.midi if el.isNote else -1)
            for el in part.flatten().notesAndRests
        )
        for part in score.parts
    )


def check_voices(score, name):
    means = []
    for part in score.parts:
        notes = list(part.flatten().notes)
        assert not any(isinstance(n, chord.Chord) for n in notes), f"{name}: chord in a part"
        means.append(sum(n.pitch.midi for n in notes) / len(notes))
    assert all(a > b for a, b in zip(means, means[1:])), \
        f"{name}: part mean pitches not strictly decreasing: {means}"


def load_report():
    report = LoadReport()
    filenames = list(corpus.chorales.Iterator(returnType="filename"))
    unique = list(dict.fromkeys(filenames))
    report.n_entries, report.n_unique = len(filenames), len(unique)

    passed = []
    for name in unique:
        score = corpus.parse(name)
        if len(score.parts) != N_VOICES:
            report.dropped.append((name, f"n_parts={len(score.parts)}"))
            continue
        el = off_grid_element(score)
        if el is not None:
            report.dropped.append(
                (name, f"off_grid offset={float(el.offset)} quarterLength={float(el.quarterLength)}"))
            continue
        check_voices(score, name)
        passed.append((name, score))
    report.n_passed_checks = len(passed)

    seen = {}
    for name, score in passed:
        key = content_key(score)
        if key in seen:
            report.dropped.append((name, f"duplicate_of={seen[key]}"))
            continue
        seen[key] = name
        report.kept.append((chorale_id(name), score))
    return report


def write_dropped(dropped, path=PROCESSED_DIR / "dropped.csv"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "reason"])
        w.writerows(dropped)


def load_chorales(write_csv=True):
    """Return the cleaned list of (chorale_id, score)."""
    report = load_report()
    if write_csv:
        write_dropped(report.dropped)
    return report.kept

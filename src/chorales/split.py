"""Train/val/test split by melody group, so the same hymn tune never lands in two splits."""
import json
import random
from collections import defaultdict
from pathlib import Path

from chorales.encode import encode
from chorales.vocab import N_VOICES, is_pitch

SEED = 413
FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}


def melody_key(ids):
    """Soprano interval sequence (transposition-invariant).

    Taken from the encoded ids, so tied notes count once and rests are skipped.
    """
    soprano = [t for t in ids[1:-1:N_VOICES] if is_pitch(t)]
    return tuple(b - a for a, b in zip(soprano, soprano[1:]))


def melody_groups(encoded):
    """{melody key: [chorale ids]} from {chorale_id: encoded dict}."""
    groups = defaultdict(list)
    for cid, e in encoded.items():
        groups[melody_key(e["ids"])].append(cid)
    return dict(groups)


def make_splits(encoded, seed=SEED):
    """Shuffle melody groups with a fixed seed and assign them so split sizes track 80/10/10."""
    groups = list(melody_groups(encoded).values())
    random.Random(seed).shuffle(groups)
    n = sum(len(g) for g in groups)
    targets = {"train": round(FRACTIONS["train"] * n), "val": round(FRACTIONS["val"] * n)}
    targets["test"] = n - targets["train"] - targets["val"]
    splits = {name: [] for name in FRACTIONS}
    for g in groups:
        # the split furthest below its target takes the whole group
        name = max(splits, key=lambda s: targets[s] - len(splits[s]))
        splits[name].extend(g)
    return {**splits, "seed": seed}


def build_splits(chorales, seed=SEED):
    return make_splits({cid: encode(s) for cid, s in chorales}, seed)


def save_splits(splits, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(splits, indent=2) + "\n")


def load_splits(path):
    return json.loads(Path(path).read_text())

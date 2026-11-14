"""PyTorch Dataset, batching and generation-time features.

Reads data/processed/{encoded,splits,ranges}.json, written by build_processed()
(run `python scripts/build_dataset.py`).
"""
import json
import random
from pathlib import Path

import torch
from music21 import meter
from torch.utils.data import Dataset

from chorales.augment import AUGMENT_MODES, allowed_shifts, compute_ranges, load_ranges, save_ranges, transpose
from chorales.encode import encode
from chorales.load import PROCESSED_DIR, load_report, write_dropped
from chorales.split import make_splits, save_splits
from chorales.vocab import BEAT_POS_SPECIAL, N_VOICES, PAD, STEPS_PER_QUARTER, VOICE_SPECIAL, write_vocab

IGNORE_INDEX = -100
# value used to pad each feature in a batch
PAD_VALUES = {"input_ids": PAD, "target_ids": IGNORE_INDEX, "voice": VOICE_SPECIAL,
              "beat_pos": BEAT_POS_SPECIAL, "fermata": 0, "onset": 0}
INPUT_FEATURES = ("voice", "beat_pos", "fermata", "onset")


def time_signature_of(score):
    ts = score.parts[0].recurse().getElementsByClass(meter.TimeSignature)
    return ts[0].ratioString if ts else "4/4"


def build_processed(out_dir=PROCESSED_DIR):
    """Load + clean + encode + split + ranges; write everything the Dataset needs."""
    out_dir = Path(out_dir)
    report = load_report()
    write_dropped(report.dropped, out_dir / "dropped.csv")
    write_vocab(out_dir / "vocab.json")
    encoded = {}
    for cid, score in report.kept:
        print("cid", cid)
        print("score", score)
        e = encode(score)
        e["time_signature"] = time_signature_of(score)
        encoded[cid] = e
    splits = make_splits(encoded)
    save_splits(splits, out_dir / "splits.json")
    save_ranges(compute_ranges(encoded[c]["ids"] for c in splits["train"]), out_dir / "ranges.json")
    (out_dir / "encoded.json").write_text(json.dumps(encoded, separators=(",", ":")))
    return report, encoded, splits


def window_starts_eval(n, max_len):
    """Consecutive windows for evaluation: every target position is predicted exactly once.

    Window i has inputs ids[s_i : s_{i+1}] and targets ids[s_i+1 : s_{i+1}+1]. s_0 = 0 and later
    starts are step boundaries (1 + 4k), at most max_len apart.
    """
    starts = [0]
    while starts[-1] + max_len < n - 1:
        s = starts[-1] + max_len
        starts.append(s - (s - 1) % N_VOICES)  # round down to 1 + 4k
    return starts


class ChoraleDataset(Dataset):
    """One item = one chorale (or window of one), as next-token-prediction tensors.

    split:        "train", "val" or "test"
    max_len:      max number of input tokens per item
    augment_mode: "sample" (one random allowed transposition each time a chorale is drawn) or
                  "all" (every allowed transposition is its own item). Ignored unless train=True.
    train:        True = random transposition + random step-aligned window for long chorales.
                  False = no transposition; long chorales become consecutive windows.
    """

    def __init__(self, split, max_len=1024, augment_mode="sample", train=None,
                 data_dir=PROCESSED_DIR, seed=None):
        if augment_mode not in AUGMENT_MODES:
            raise ValueError(f"augment_mode must be one of {AUGMENT_MODES}")
        data_dir = Path(data_dir)
        encoded_path = data_dir / "encoded.json"
        if not encoded_path.exists():
            raise FileNotFoundError(f"{encoded_path} missing; run `python scripts/build_dataset.py`")
        encoded = json.loads(encoded_path.read_text())
        ids_in_split = json.loads((data_dir / "splits.json").read_text())[split]
        self.ranges = load_ranges(data_dir / "ranges.json")
        self.split, self.max_len, self.augment_mode = split, max_len, augment_mode
        self.train = (split == "train") if train is None else train
        self.rng = random.Random(seed)
        self.chorales = {cid: encoded[cid] for cid in ids_in_split}

        # items: (chorale_id, shift or None for "sample a shift", window start or None for random)
        self.items = []
        for cid, e in self.chorales.items():
            n = len(e["ids"])
            if self.train:
                shifts = allowed_shifts(e["ids"], self.ranges)
                e["shifts"] = shifts
                for k in (shifts if augment_mode == "all" else [None]):
                    self.items.append((cid, k, None))
            else:
                for s in window_starts_eval(n, max_len):
                    self.items.append((cid, 0, s))

    def __len__(self):
        return len(self.items)

    def _train_window_start(self, n):
        if n <= self.max_len + 1:
            return 0
        last = n - (self.max_len + 1)
        starts = [0] + list(range(1, last + 1, N_VOICES))
        return self.rng.choice(starts)

    def __getitem__(self, i):
        cid, shift, start = self.items[i]
        e = self.chorales[cid]
        n = len(e["ids"])
        if shift is None:
            shift = self.rng.choice(e["shifts"])
        if start is None:
            start = self._train_window_start(n)
            stop = min(n, start + self.max_len + 1)
        else:
            starts = window_starts_eval(n, self.max_len)
            nxt = starts.index(start) + 1
            stop = starts[nxt] + 1 if nxt < len(starts) else n
        ids = transpose(e["ids"][start:stop], shift) if shift else e["ids"][start:stop]
        item = {"input_ids": ids[:-1], "target_ids": ids[1:]}
        for k in INPUT_FEATURES:
            item[k] = e[k][start:stop - 1]
        out = {k: torch.tensor(v, dtype=torch.long) for k, v in item.items()}
        out["chorale_id"], out["shift"], out["start"] = cid, shift, start
        return out


def collate_fn(batch):
    """Pad to the longest item. pad_mask is True at padded positions (PyTorch key_padding_mask
    convention); target_ids are IGNORE_INDEX (-100) there."""
    length = max(len(b["input_ids"]) for b in batch)
    out = {}
    for k, pad in PAD_VALUES.items():
        t = torch.full((len(batch), length), pad, dtype=torch.long)
        for row, b in enumerate(batch):
            t[row, :len(b[k])] = b[k]
        out[k] = t
    lengths = torch.tensor([len(b["input_ids"]) for b in batch])
    out["pad_mask"] = torch.arange(length)[None, :] >= lengths[:, None]
    out["lengths"] = lengths
    out["chorale_id"] = [b["chorale_id"] for b in batch]
    out["shift"] = torch.tensor([b["shift"] for b in batch])
    return out


def generation_features(position, time_signature="4/4", fermata_template=None, start_beat_pos=0):
    """voice, beat_pos, fermata for sequence position `position` (0 = <START>) during sampling.

    start_beat_pos: 16th position within the bar of step 0 (e.g. 8 for BWV 269's one-beat pickup
                    in 3/4); 0 = the piece starts on a downbeat.
    fermata_template: per-step 0/1 list (e.g. copied from a real chorale); steps past its end get 0.
    """
    if position == 0:
        return {"voice": VOICE_SPECIAL, "beat_pos": BEAT_POS_SPECIAL, "fermata": 0}
    step = (position - 1) // N_VOICES
    bar_steps = int(meter.TimeSignature(time_signature).barDuration.quarterLength * STEPS_PER_QUARTER)
    fermata = int(fermata_template[step]) if fermata_template is not None and step < len(fermata_template) else 0
    return {"voice": (position - 1) % N_VOICES,
            "beat_pos": (start_beat_pos + step) % bar_steps,
            "fermata": fermata}

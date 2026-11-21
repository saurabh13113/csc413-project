"""Range-respecting transposition augmentation.

Ranges are computed on the TRAIN split only and saved to ranges.json; the rule checker's voice
range rule must read the same file.
"""
import json
from pathlib import Path

from chorales.vocab import MAX_PITCH_ID, MIN_PITCH_ID, N_VOICES, id_to_pitch, is_pitch

VOICE_NAMES = ("S", "A", "T", "B")
SHIFTS = range(-6, 6)  # -6..+5 semitones: each of the 12 keys once
AUGMENT_MODES = ("all", "sample")


def voice_pitches(ids):
    """MIDI pitches per voice, from an encoded id list."""
    out = [[] for _ in range(N_VOICES)]
    for p, t in enumerate(ids[1:-1]):
        if is_pitch(t):
            out[p % N_VOICES].append(id_to_pitch(t))
    return out


def compute_ranges(id_lists):
    """{voice name: [min MIDI, max MIDI]} over the given encoded sequences."""
    lo, hi = [999] * N_VOICES, [-1] * N_VOICES
    for ids in id_lists:
        for v, pitches in enumerate(voice_pitches(ids)):
            if pitches:
                lo[v], hi[v] = min(lo[v], *pitches), max(hi[v], *pitches)
    return {name: [lo[v], hi[v]] for v, name in enumerate(VOICE_NAMES)}


def save_ranges(ranges, path, source="train split"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source, "midi": ranges}, indent=2) + "\n")


def load_ranges(path):
    return json.loads(Path(path).read_text())["midi"]


def allowed_shifts(ids, ranges):
    """Semitone shifts k in -6..+5 that keep every voice inside its range."""
    shifts = []
    per_voice = voice_pitches(ids)
    for k in SHIFTS:
        ok = all(
            not pitches or (ranges[name][0] <= min(pitches) + k and max(pitches) + k <= ranges[name][1])
            for name, pitches in zip(VOICE_NAMES, per_voice)
        )
        if ok:
            shifts.append(k)
    return shifts


def transpose(ids, k):
    """Add k to every pitch ID; HOLD/REST/specials unchanged."""
    out = [t + k if is_pitch(t) else t for t in ids]
    bad = [t for t, orig in zip(out, ids) if is_pitch(orig) and not MIN_PITCH_ID <= t <= MAX_PITCH_ID]
    if bad:
        raise ValueError(f"transposing by {k} leaves the vocabulary: {bad[:5]}")
    return out

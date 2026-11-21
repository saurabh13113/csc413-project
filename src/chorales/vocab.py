"""Token vocabulary (51 tokens) and per-token feature constants."""
import json
from pathlib import Path

PAD, START, END, HOLD, REST = 0, 1, 2, 3, 4
SPECIAL_NAMES = {PAD: "<PAD>", START: "<START>", END: "<END>", HOLD: "HOLD", REST: "REST"}

MIN_PITCH, MAX_PITCH = 36, 81          # MIDI range observed in the cleaned data
PITCH_OFFSET = 31                      # pitch ID = MIDI - 31
MIN_PITCH_ID = MIN_PITCH - PITCH_OFFSET  # 5
MAX_PITCH_ID = MAX_PITCH - PITCH_OFFSET  # 50
VOCAB_SIZE = MAX_PITCH_ID + 1          # 51

N_VOICES = 4
VOICE_SPECIAL = 4                      # voice feature for <START>/<END>/<PAD>
BEAT_POS_SPECIAL = 24                  # beat_pos feature for specials (positions are 0..23)
STEPS_PER_QUARTER = 4                  # 16th-note grid


def pitch_to_id(midi):
    if not MIN_PITCH <= midi <= MAX_PITCH:
        raise ValueError(f"MIDI pitch {midi} outside {MIN_PITCH}..{MAX_PITCH}")
    return midi - PITCH_OFFSET


def id_to_pitch(token):
    if not is_pitch(token):
        raise ValueError(f"token {token} is not a pitch")
    return token + PITCH_OFFSET


def is_pitch(token):
    return MIN_PITCH_ID <= token <= MAX_PITCH_ID


def token_name(token):
    return SPECIAL_NAMES.get(token) or str(id_to_pitch(token))


def vocab_dict():
    """{token string: id}; pitches are written as their MIDI number."""
    d = {name: i for i, name in sorted(SPECIAL_NAMES.items())}
    d.update({str(m): pitch_to_id(m) for m in range(MIN_PITCH, MAX_PITCH + 1)})
    return d


def write_vocab(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab_dict(), indent=2) + "\n")

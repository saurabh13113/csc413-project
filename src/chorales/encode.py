"""Encode a 4-part chorale into interleaved S/A/T/B tokens on a 16th-note grid, and decode back.

Sequence layout: <START>, then for each 16th step t the four voice tokens S A T B, then <END>.
Slot rule for voice v at step t:
  rest sounding                                  -> REST (every step of the rest)
  note starting at t, not a tie continuation     -> pitch ID
  otherwise (note still sounding / tied into t)  -> HOLD
Grace notes (zero duration) are skipped.
"""
from fractions import Fraction
from pathlib import Path

from music21 import expressions, meter, note, stream

from chorales.vocab import (BEAT_POS_SPECIAL, END, HOLD, N_VOICES, PAD, REST, START,
                            STEPS_PER_QUARTER, VOICE_SPECIAL, id_to_pitch, is_pitch,
                            pitch_to_id)

FEATURES = ("ids", "voice", "beat_pos", "fermata", "onset")


def to_steps(quarter_length):
    steps = Fraction(quarter_length) * STEPS_PER_QUARTER
    if steps.denominator != 1:
        raise ValueError(f"{quarter_length} quarter notes is not on the 16th-note grid")
    return int(steps)


def is_tie_continuation(n):
    return n.tie is not None and n.tie.type in ("stop", "continue")


def has_fermata(el):
    return any(isinstance(e, expressions.Fermata) for e in el.expressions)


def voice_slots(part, n_steps):
    """Token ID and fermata flag for each step of one voice."""
    ids = [None] * n_steps
    fermata = [0] * n_steps
    for el in part.flatten().notesAndRests:
        if el.quarterLength == 0:  # grace note
            continue
        start = to_steps(el.offset)
        end = start + to_steps(el.quarterLength)
        for t in range(start, end):
            if el.isRest:
                ids[t] = REST
            elif t == start and not is_tie_continuation(el):
                ids[t] = pitch_to_id(el.pitch.midi)
            else:
                ids[t] = HOLD
            fermata[t] = int(has_fermata(el))
    missing = [t for t, x in enumerate(ids) if x is None]
    if missing:
        raise ValueError(f"part {part.partName!r} has no note or rest at steps {missing[:5]}...")
    return ids, fermata


def beat_positions(score, n_steps):
    """16th-note position within the bar for each step, from the measure structure.

    position = (measure.paddingLeft + offset in measure) * 4, so a pickup bar starts mid-bar.
    Wrapped modulo the bar length, because a few measures are longer than their time signature
    (bwv324 has a 10-beat measure in 4/4).
    """
    pos = [None] * n_steps
    for m in score.parts[0].getElementsByClass(stream.Measure):
        bar_steps = to_steps(m.barDuration.quarterLength)
        first = to_steps(m.offset)
        pad = to_steps(m.paddingLeft)
        for k in range(to_steps(m.duration.quarterLength)):
            if first + k < n_steps:
                pos[first + k] = (pad + k) % bar_steps
    if any(p is None for p in pos):
        raise ValueError("measures don't cover every step")
    return pos


def encode(score):
    """Return a dict of equal-length int lists: ids, voice, beat_pos, fermata, onset."""
    parts = list(score.parts)
    print("parts",parts)
    if len(parts) != N_VOICES:
        raise ValueError(f"expected {N_VOICES} parts, got {len(parts)}")
    n_steps = to_steps(score.highestTime)
    slots = [voice_slots(p, n_steps) for p in parts]
    soprano_fermata = slots[0][1]
    beat_pos = beat_positions(score, n_steps)

    out = {k: [] for k in FEATURES}

    def add(token, voice, pos, ferm):
        out["ids"].append(token)
        out["voice"].append(voice)
        out["beat_pos"].append(pos)
        out["fermata"].append(ferm)
        out["onset"].append(int(token == REST or is_pitch(token)))

    add(START, VOICE_SPECIAL, BEAT_POS_SPECIAL, 0)
    for t in range(n_steps):
        for v in range(N_VOICES):
            add(slots[v][0][t], v, beat_pos[t], soprano_fermata[t])
    add(END, VOICE_SPECIAL, BEAT_POS_SPECIAL, 0)
    return out


def decode(ids, time_signature="4/4"):
    """Turn a token sequence back into a 4-part music21 Score.

    <START>/<END>/<PAD> are stripped. A trailing incomplete step is ignored. A HOLD with no
    note sounding (only possible in sequences that break the format grammar) becomes a rest.
    """
    body = [t for t in ids if t not in (START, END, PAD)]
    n_steps = len(body) // N_VOICES
    body = body[:n_steps * N_VOICES]
    dur = Fraction(1, STEPS_PER_QUARTER)

    score = stream.Score()
    for v, name in enumerate(["Soprano", "Alto", "Tenor", "Bass"]):
        part = stream.Part(id=name)
        part.partName = name
        part.append(meter.TimeSignature(time_signature))
        current = None  # the Note or Rest being extended
        for t in range(n_steps):
            token = body[t * N_VOICES + v]
            if is_pitch(token):
                current = note.Note(id_to_pitch(token), quarterLength=dur)
                part.append(current)
            elif token == HOLD and isinstance(current, note.Note):
                current.quarterLength += dur
            elif isinstance(current, note.Rest):  # REST (or orphan HOLD) continuing a rest
                current.quarterLength += dur
            else:
                current = note.Rest(quarterLength=dur)
                part.append(current)
        # Add bar lines; notes crossing a bar line become tied notes (they re-encode as HOLD).
        part.makeMeasures(inPlace=True)
        part.makeTies(inPlace=True)
        score.insert(0, part)
    return score


def write_midi(ids, path, time_signature="4/4"):
    """Decode a token sequence and save it as a MIDI file. Returns the path."""
    score = decode(ids, time_signature)
    return Path(score.write("midi", fp=str(path)))

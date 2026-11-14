"""Format grammar for generation: which token IDs may come next.

These are format rules only (a valid token sequence). Music rules (voice ranges, parallel
fifths/octaves, voice crossing) live in the rule checker and are applied on top.
"""
from chorales.vocab import END, HOLD, MAX_PITCH_ID, MIN_PITCH_ID, N_VOICES, REST, START, is_pitch

PITCH_IDS = frozenset(range(MIN_PITCH_ID, MAX_PITCH_ID + 1))


def next_voice(ids_so_far):
    """Voice (0=S .. 3=B) of the next slot, given a sequence that starts with <START>."""
    return (len(ids_so_far) - 1) % N_VOICES


def allowed_tokens(ids_so_far):
    """Set of token IDs allowed at the next position.

    ids_so_far must start with <START>. Never <START> or <PAD>. Nothing is allowed after <END>.
    HOLD only if the next slot's voice has a note sounding (its previous token is a pitch or
    HOLD). <END> only when the next slot would be a soprano slot, after at least one full step.
    """
    if not ids_so_far or ids_so_far[0] != START:
        raise ValueError("sequence must start with <START>")
    if ids_so_far[-1] == END:
        return set()

    allowed = set(PITCH_IDS) | {REST}
    p = len(ids_so_far)
    prev_same_voice = ids_so_far[p - N_VOICES] if p - N_VOICES >= 1 else None
    if prev_same_voice is not None and (prev_same_voice == HOLD or is_pitch(prev_same_voice)):
        allowed.add(HOLD)
    if next_voice(ids_so_far) == 0 and p > 1:
        allowed.add(END)
    return allowed

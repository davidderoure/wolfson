"""
Token encoding for pitch + duration sequences.

Sequence format (interleaved):
    PITCH_0, DUR_0, PITCH_1, DUR_1, ..., END

Pitch tokens:    0 .. N_PITCHES-1
Duration tokens: N_PITCHES .. N_PITCHES+N_DUR_BUCKETS-1
END token:       N_PITCHES + N_DUR_BUCKETS

Duration is beat-relative (duration_sec / beat_dur_sec), so it is
tempo-independent and preserves expressive timing. Bucket boundaries are
log-spaced from MIN_DUR_BEATS to MAX_DUR_BEATS.

Chord conditioning uses a SEPARATE vocabulary (not part of the output tokens).
Chord indices are defined in data/chords.py. The model takes chord as an
additional input embedding at each timestep but does not predict it as output.
"""

import numpy as np

PITCH_MIN = 44          # Ab2 — covers all sax voices
PITCH_MAX = 93          # A6
N_PITCHES = PITCH_MAX - PITCH_MIN + 1   # 50

# Duration buckets: log-spaced from a 32nd note to 4 beats
MIN_DUR_BEATS = 0.03125  # 1/32 beat
MAX_DUR_BEATS = 4.0
N_DUR_BUCKETS = 32

_DUR_BOUNDARIES = np.geomspace(MIN_DUR_BEATS, MAX_DUR_BEATS, N_DUR_BUCKETS + 1)

PITCH_OFFSET = 0
DUR_OFFSET   = N_PITCHES                    # 50
END_TOKEN    = N_PITCHES + N_DUR_BUCKETS    # 82
VOCAB_SIZE   = END_TOKEN + 1                # 83


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def pitch_to_token(midi_pitch: int) -> int:
    p = int(np.clip(midi_pitch, PITCH_MIN, PITCH_MAX))
    return p - PITCH_MIN + PITCH_OFFSET


def token_to_pitch(token: int) -> int:
    return token - PITCH_OFFSET + PITCH_MIN


def dur_to_token(duration_beats: float) -> int:
    dur    = np.clip(duration_beats, MIN_DUR_BEATS, MAX_DUR_BEATS)
    bucket = int(np.searchsorted(_DUR_BOUNDARIES, dur, side="right")) - 1
    bucket = int(np.clip(bucket, 0, N_DUR_BUCKETS - 1))
    return bucket + DUR_OFFSET


def token_to_dur(token: int) -> float:
    """Geometric midpoint of the duration bucket, in beats."""
    bucket = int(np.clip(token - DUR_OFFSET, 0, N_DUR_BUCKETS - 1))
    lo = _DUR_BOUNDARIES[bucket]
    hi = _DUR_BOUNDARIES[bucket + 1]
    return float(np.sqrt(lo * hi))


def is_pitch_token(token: int) -> bool:
    return PITCH_OFFSET <= token < PITCH_OFFSET + N_PITCHES


def is_dur_token(token: int) -> bool:
    return DUR_OFFSET <= token < DUR_OFFSET + N_DUR_BUCKETS


# ---------------------------------------------------------------------------
# Phrase <-> token sequence conversion
# ---------------------------------------------------------------------------

def phrase_to_tokens(phrase: list[dict], tempo_bpm: float = 120.0) -> list[int]:
    """
    Convert a phrase to: [PITCH, DUR, PITCH, DUR, ..., END].

    Note dicts must have: pitch (int), onset (float, sec), offset (float, sec).
    Optional per-note: beat_dur_sec (float) — local beat duration. Falls back
    to 60/tempo_bpm if absent.
    """
    if not phrase:
        return [END_TOKEN]

    fallback_beat_dur = 60.0 / tempo_bpm
    tokens = []
    for note in phrase:
        beat_dur       = note.get("beat_dur_sec") or fallback_beat_dur
        duration_beats = (note["offset"] - note["onset"]) / beat_dur
        tokens.append(pitch_to_token(note["pitch"]))
        tokens.append(dur_to_token(duration_beats))
    tokens.append(END_TOKEN)
    return tokens


def phrase_to_chord_sequence(phrase: list[dict]) -> list[int]:
    """
    Return a chord index per token position in the token sequence.
    Length matches phrase_to_tokens output (2*n_notes + 1).

    Each note dict may have chord_idx (int). The same chord index is repeated
    for the pitch token and its paired duration token. END position uses NC.
    """
    from data.chords import NC_INDEX
    chord_seq = []
    for note in phrase:
        idx = note.get("chord_idx", NC_INDEX)
        chord_seq.append(idx)   # pitch token position
        chord_seq.append(idx)   # duration token position
    chord_seq.append(NC_INDEX)  # END token position
    return chord_seq


def tokens_to_phrase(tokens: list[int]) -> list[dict]:
    """
    Reconstruct {pitch, duration_beats} dicts from a token sequence.
    Stops at END_TOKEN.
    """
    notes = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if t == END_TOKEN:
            break
        if is_pitch_token(t) and i + 1 < len(tokens) and is_dur_token(tokens[i + 1]):
            notes.append({
                "pitch":          token_to_pitch(t),
                "duration_beats": token_to_dur(tokens[i + 1]),
            })
            i += 2
        else:
            i += 1
    return notes

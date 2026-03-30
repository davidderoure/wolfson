"""
Generates sax response phrases using the LSTM model.

Bias layers (applied as logit additions during sampling):
  - Chord conditioning     : chord_idx passed at every timestep
  - Pitch range            : soft penalty outside E3–E6 sax register
  - Scale pitch bias       : positive bias for in-scale pitch classes
  - Contour steering       : distance-from-mean push toward ascending/descending
  - Swing/triplet bias     : positive bias on 12/8 triplet-grid duration tokens
  - Energy arc             : position-dependent pitch + duration shaping
      'arch'     — rise to midpoint, fall to end
      'ramp_up'  — ascending energy throughout
      'ramp_down'— descending energy throughout
      'spike'    — sharp burst peaking at ~30% through
      'flat'     — no arc (default)
  - Motivic development    : bias toward continuing recognised interval patterns
  - Voice leading          : chord-tone targeting at phrase end + stepwise preference

Bias budget
-----------
Biases are calibrated so their combined effect at any step stays musical:
  contour         max ~2.5 logits (1.5 when energy arc active)
  energy arc      max ~1.5 logits pitch + 0.8 dur
  voice leading   max ~1.0 logits (grows with arc_position)
  motif           2.0 logits on one specific token (sparse, fires rarely)
  scale           2.0 logits (flat on all in-scale tokens)
  stepwise        0.4 logits on ±1–2 semitone tokens
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import (
    phrase_to_tokens, phrase_to_chord_sequence, tokens_to_phrase,
    END_TOKEN, VOCAB_SIZE, DUR_OFFSET, N_PITCHES, N_DUR_BUCKETS,
    dur_to_token, token_to_pitch, pitch_to_token,
)
from data.chords import NC_INDEX, CHORD_VOCAB_SIZE
from data.scales import chord_tones
from generator.lstm_model import PhraseModel
from config import MAX_GENERATED_NOTES, GENERATION_TEMPERATURE, DEFAULT_INSTRUMENT

MODELS_DIR = Path(__file__).parent.parent / "models"

# ---------------------------------------------------------------------------
# Bias strength constants
# ---------------------------------------------------------------------------

# Contour steering — full strength when no energy arc, reduced when arc active
CONTOUR_STEER_ONSET            = 0.0
CONTOUR_BIAS_STRENGTH          = 2.5
CONTOUR_BIAS_STRENGTH_WITH_ARC = 1.5   # reduced to share budget with energy arc

# Swing / triplet bias
TRIPLET_DUR_BEATS   = [1/3, 2/3, 1.0, 4/3, 2.0]
SWING_BIAS_STRENGTH = 1.2

# Scale pitch bias
SCALE_BIAS_STRENGTH = 2.0

# Pitch range soft limits
PITCH_RANGE_MIN      = 52    # E3
PITCH_RANGE_MAX      = 88    # E6
PITCH_RANGE_STRENGTH = 2.0   # logit penalty per semitone outside range

# Minimum duration floor
MIN_DURATION_BEATS = 0.2     # ~100 ms at 120 BPM

# Energy arc — internal phrase shaping
ENERGY_PITCH_STRENGTH = 1.5  # max pitch logit bias from arc
ENERGY_DUR_STRENGTH   = 0.8  # max duration logit bias from arc

# Motivic development
MOTIF_BIAS_STRENGTH   = 2.0  # logit bias for motif-continuation pitch token

# Voice leading
VOICE_LEADING_STRENGTH = 0.6  # max chord-tone boost (scales with arc_position)
                               # reduced from 1.0 — prevents over-sticky cadences
STEPWISE_BIAS_STRENGTH = 0.4  # gentle bias toward ±1–2 semitone motion

# Repetition penalty — discourages consecutive repeated notes
# Grows with the number of consecutive repeats so an occasional same-note
# passing tone is allowed but long runs are strongly suppressed.
REPEAT_PENALTY_BASE    = 1.5  # logit penalty for first immediate repeat
REPEAT_PENALTY_SCALE   = 1.0  # additional penalty per further consecutive repeat


class PhraseGenerator:
    def __init__(self, instrument=None, model_path=None):
        instrument = instrument or DEFAULT_INSTRUMENT
        self.model = PhraseModel()

        if model_path is None:
            model_path = MODELS_DIR / f"{instrument}_best.pt"

        if Path(model_path).exists():
            ckpt = torch.load(model_path, map_location="cpu")
            state = ckpt["model"] if "model" in ckpt else ckpt
            self.model.load_state_dict(state)
            print(f"Loaded model: {model_path}")
        else:
            print(f"No model at {model_path} — using untrained weights (random output).")

        self.model.eval()

    def generate(
        self,
        seed_phrase:         list[dict],
        tempo_bpm:           float      = 120.0,
        n_notes:             int        = None,
        temperature:         float      = None,
        contour_target:      str        = "neutral",
        chord_idx:           int        = NC_INDEX,
        swing_bias:          float      = 0.0,
        scale_pitch_classes: frozenset  = None,
        phrase_energy_arc:   str        = "flat",
        motif_targets:       list       = None,
        motif_strength:      float      = 0.0,
    ) -> list[dict]:
        """
        Generate a sax phrase seeded by a bass phrase.

        seed_phrase         list of note dicts (pitch, onset, offset, [beat_dur_sec])
        tempo_bpm           fallback tempo if notes lack beat_dur_sec
        contour_target      'ascending' | 'descending' | 'neutral'
        chord_idx           current chord token (NC_INDEX if unknown)
        swing_bias          0–1 strength of triplet-grid duration bias
        scale_pitch_classes frozenset of pitch classes (0–11) for scale bias
        phrase_energy_arc   internal energy shape: 'arch'|'ramp_up'|'ramp_down'|
                            'spike'|'flat'
        motif_targets       list of interval tuples to lean toward, e.g. [(2,-1,5)]
        motif_strength      0–1 scale on MOTIF_BIAS_STRENGTH

        Returns list of {pitch, duration_beats, velocity_scale} dicts.
        velocity_scale is 0.75–1.25; apply to base phrase velocity in the caller.
        """
        n_notes       = n_notes     or MAX_GENERATED_NOTES
        temperature   = temperature or GENERATION_TEMPERATURE
        motif_targets = motif_targets or []
        arc_active    = phrase_energy_arc != "flat"

        # Seed mean pitch — contour reference for the first generated note
        seed_pitches    = [n["pitch"] for n in seed_phrase if "pitch" in n]
        seed_mean_pitch = sum(seed_pitches) / len(seed_pitches) if seed_pitches else 69.0

        seed_tokens = phrase_to_tokens(seed_phrase, tempo_bpm)
        seed_chords = phrase_to_chord_sequence(seed_phrase)

        if chord_idx != NC_INDEX:
            seed_chords = [chord_idx] * len(seed_chords)

        tok_tensor   = torch.tensor([seed_tokens], dtype=torch.long)
        chord_tensor = torch.tensor([seed_chords], dtype=torch.long)

        generated_tokens    = []
        hidden              = None
        last_pitch_token    = -1   # most recent generated pitch token; -1 = none yet
        consecutive_repeats = 0    # how many times last_pitch_token has been repeated

        # Reduce contour strength when energy arc is also active (shared budget)
        contour_strength = (
            CONTOUR_BIAS_STRENGTH_WITH_ARC if arc_active else CONTOUR_BIAS_STRENGTH
        )

        with torch.no_grad():
            # Prime the hidden state with the full seed sequence
            logits, hidden = self.model(tok_tensor, chord_tensor, hidden)

            last_tok   = torch.tensor([[seed_tokens[-1]]], dtype=torch.long)
            last_chord = torch.tensor([[chord_idx]],       dtype=torch.long)
            expecting  = "pitch"
            note_count = 0

            for step in range(n_notes * 2):
                logits, hidden = self.model(last_tok, last_chord, hidden)
                tok_logits = logits[0, -1, :] / temperature

                # Hard alternation mask: force PITCH → DUR → PITCH → …
                tok_logits = tok_logits + _alternation_mask(expecting)

                # Arc position: 0.0 at first note, 1.0 at last
                arc_position = note_count / max(n_notes, 1)

                if expecting == "pitch":
                    # Pitch-range soft limits (precomputed)
                    tok_logits = tok_logits + _PITCH_RANGE_BIAS

                    # Scale / mode bias
                    if scale_pitch_classes:
                        tok_logits = _apply_scale_bias(tok_logits, scale_pitch_classes)

                    # Contour steering (from note 0)
                    if note_count >= int(n_notes * CONTOUR_STEER_ONSET):
                        tok_logits = _apply_contour_bias(
                            tok_logits, generated_tokens, contour_target,
                            seed_mean_pitch, contour_strength,
                        )

                    # Energy arc — position-dependent pitch push
                    if arc_active:
                        tok_logits = _apply_energy_pitch_bias(
                            tok_logits, arc_position, phrase_energy_arc,
                            seed_mean_pitch,
                        )

                    # Motivic development — continue recognised interval patterns
                    if motif_targets and motif_strength > 0.0:
                        tok_logits = _apply_motif_bias(
                            tok_logits, generated_tokens,
                            motif_targets, motif_strength,
                        )

                    # Voice leading — chord-tone endpoint bias + stepwise preference
                    tok_logits = _apply_voice_leading_bias(
                        tok_logits, arc_position, chord_idx, last_pitch_token,
                    )

                    # Repetition penalty — suppress running on the same note
                    if last_pitch_token >= 0 and consecutive_repeats > 0:
                        tok_logits = _apply_repeat_penalty(
                            tok_logits, last_pitch_token, consecutive_repeats,
                        )

                elif expecting == "duration":
                    # Triplet/swing bias
                    if swing_bias > 0.0:
                        tok_logits = _apply_swing_bias(tok_logits, swing_bias)

                    # Energy arc — position-dependent duration push
                    if arc_active:
                        tok_logits = _apply_energy_dur_bias(
                            tok_logits, arc_position, phrase_energy_arc,
                        )

                probs = F.softmax(tok_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()

                if token == END_TOKEN:
                    break

                generated_tokens.append(token)
                last_tok   = torch.tensor([[token]],     dtype=torch.long)
                last_chord = torch.tensor([[chord_idx]], dtype=torch.long)

                if expecting == "pitch":
                    if token == last_pitch_token:
                        consecutive_repeats += 1
                    else:
                        consecutive_repeats  = 0
                    last_pitch_token = token
                    note_count += 1
                    expecting = "duration"
                else:
                    expecting = "pitch"

        notes = tokens_to_phrase(generated_tokens)
        n = len(notes)

        for i, note in enumerate(notes):
            # Minimum duration floor
            note["duration_beats"] = max(note["duration_beats"], MIN_DURATION_BEATS)

            # Per-note velocity scale from energy arc (0.75 quiet → 1.25 loud)
            if arc_active and n > 0:
                pos      = i / max(n - 1, 1)
                activity = _energy_activity_level(pos, phrase_energy_arc)
                note["velocity_scale"] = 0.75 + 0.5 * activity
            else:
                note["velocity_scale"] = 1.0

        return notes


# ---------------------------------------------------------------------------
# Precomputed tensors
# ---------------------------------------------------------------------------

def _build_triplet_token_set() -> set:
    tokens = set()
    for dur_beats in TRIPLET_DUR_BEATS:
        t = dur_to_token(dur_beats)
        tokens.add(t)
        if t - 1 >= DUR_OFFSET:
            tokens.add(t - 1)
        if t + 1 < DUR_OFFSET + N_DUR_BUCKETS:
            tokens.add(t + 1)
    return tokens

_TRIPLET_TOKENS: set = _build_triplet_token_set()


def _apply_pitch_range_bias() -> torch.Tensor:
    from data.encoding import PITCH_MIN
    bias = torch.zeros(VOCAB_SIZE)
    for tok in range(N_PITCHES):
        pitch = PITCH_MIN + tok
        if pitch < PITCH_RANGE_MIN:
            bias[tok] = -(PITCH_RANGE_MIN - pitch) * PITCH_RANGE_STRENGTH
        elif pitch > PITCH_RANGE_MAX:
            bias[tok] = -(pitch - PITCH_RANGE_MAX) * PITCH_RANGE_STRENGTH
    return bias

_PITCH_RANGE_BIAS: torch.Tensor = _apply_pitch_range_bias()

# Duration split: tokens below this index are "short" (< ~0.5 beats)
_ENERGY_DUR_SPLIT: int = dur_to_token(0.5)


# ---------------------------------------------------------------------------
# Bias helpers — existing
# ---------------------------------------------------------------------------

def _alternation_mask(expecting: str) -> torch.Tensor:
    mask = torch.full((VOCAB_SIZE,), float("-inf"))
    if expecting == "pitch":
        mask[:N_PITCHES] = 0.0
        mask[END_TOKEN]  = 0.0
    else:
        mask[DUR_OFFSET:END_TOKEN] = 0.0
    return mask


def _apply_contour_bias(
    logits:           torch.Tensor,
    generated_so_far: list,
    target:           str,
    seed_mean_pitch:  float = 69.0,
    strength:         float = CONTOUR_BIAS_STRENGTH,
) -> torch.Tensor:
    """
    Soft bias toward higher or lower pitches depending on target contour.
    Reference is the running mean of generated pitches, falling back to the
    seed mean so the first note starts in the right register.
    """
    if target == "neutral":
        return logits

    pitch_tokens = [t for t in generated_so_far if t < N_PITCHES]
    if pitch_tokens:
        mean_pitch_token = sum(pitch_tokens) / len(pitch_tokens)
    else:
        from data.encoding import PITCH_MIN
        mean_pitch_token = seed_mean_pitch - PITCH_MIN

    bias = torch.zeros(VOCAB_SIZE)
    for i in range(N_PITCHES):
        distance = i - mean_pitch_token
        if target == "descending":
            bias[i] = -distance * (strength / N_PITCHES)
        else:   # ascending / answer / question
            bias[i] = distance * (strength / N_PITCHES)

    return logits + bias


def _apply_swing_bias(logits: torch.Tensor, strength: float) -> torch.Tensor:
    bias = torch.zeros(VOCAB_SIZE)
    for t in _TRIPLET_TOKENS:
        bias[t] = SWING_BIAS_STRENGTH * strength
    return logits + bias


def _apply_scale_bias(
    logits:        torch.Tensor,
    pitch_classes: frozenset,
) -> torch.Tensor:
    if not pitch_classes:
        return logits
    from data.encoding import PITCH_MIN
    bias = torch.zeros(VOCAB_SIZE)
    for tok in range(N_PITCHES):
        if (PITCH_MIN + tok) % 12 in pitch_classes:
            bias[tok] = SCALE_BIAS_STRENGTH
    return logits + bias


# ---------------------------------------------------------------------------
# Bias helpers — energy arc
# ---------------------------------------------------------------------------

def _energy_pitch_signal(arc_position: float, arc_shape: str) -> float:
    """
    Signed directional signal for the energy arc pitch bias.
    +1.0 = bias upward, -1.0 = bias downward, 0.0 = neutral.
    """
    if arc_shape == "ramp_up":
        return arc_position
    if arc_shape == "ramp_down":
        return -(1.0 - arc_position)
    if arc_shape == "arch":
        # +1 at start → 0 at midpoint → -1 at end
        return (0.5 - arc_position) * 2.0
    if arc_shape == "spike":
        # upward burst peaking at 0.3, then fall
        return max(-1.0, (0.3 - arc_position) * 3.0)
    return 0.0


def _energy_activity_level(arc_position: float, arc_shape: str) -> float:
    """
    Scalar activity level 0.0–1.0.
    High = shorter notes, louder; low = longer notes, quieter.
    """
    if arc_shape == "flat":
        return 0.0
    if arc_shape == "ramp_up":
        return arc_position
    if arc_shape == "ramp_down":
        return 1.0 - arc_position
    if arc_shape == "arch":
        return 1.0 - abs(2.0 * arc_position - 1.0)
    if arc_shape == "spike":
        return max(0.0, 1.0 - abs(arc_position - 0.3) * 4.0)
    return 0.0


def _apply_energy_pitch_bias(
    logits:          torch.Tensor,
    arc_position:    float,
    arc_shape:       str,
    seed_mean_pitch: float,
) -> torch.Tensor:
    """
    Position-dependent pitch push based on energy arc shape.
    Uses the same distance-from-mean approach as contour bias.
    """
    signal = _energy_pitch_signal(arc_position, arc_shape)
    if signal == 0.0:
        return logits

    from data.encoding import PITCH_MIN
    mean_tok = seed_mean_pitch - PITCH_MIN
    bias = torch.zeros(VOCAB_SIZE)
    for i in range(N_PITCHES):
        distance = i - mean_tok
        bias[i]  = distance * signal * (ENERGY_PITCH_STRENGTH / N_PITCHES)

    return logits + bias


def _apply_energy_dur_bias(
    logits:       torch.Tensor,
    arc_position: float,
    arc_shape:    str,
) -> torch.Tensor:
    """
    Position-dependent duration push: high activity → shorter notes.
    Short tokens (< ~0.5 beats) are boosted; long tokens are gently penalised.
    """
    activity = _energy_activity_level(arc_position, arc_shape)
    if activity == 0.0:
        return logits

    bias  = torch.zeros(VOCAB_SIZE)
    split = _ENERGY_DUR_SPLIT

    for tok in range(DUR_OFFSET, END_TOKEN):
        if tok < split:
            bias[tok] =  ENERGY_DUR_STRENGTH * activity          # short: boost
        else:
            bias[tok] = -ENERGY_DUR_STRENGTH * activity * 0.5    # long: gentle pull

    return logits + bias


# ---------------------------------------------------------------------------
# Bias helpers — motivic development
# ---------------------------------------------------------------------------

def _apply_motif_bias(
    logits:           torch.Tensor,
    generated_tokens: list,
    motif_targets:    list,
    strength:         float,
) -> torch.Tensor:
    """
    Bias toward the next pitch that would continue a recognised interval motif.

    For each motif in motif_targets (a list of signed-semitone interval tuples),
    check whether the most recently generated pitches match a prefix of the motif.
    If so, add MOTIF_BIAS_STRENGTH * strength to the token that would complete the
    next step.

    Only the longest matching prefix fires, preventing double-counting.
    """
    pitch_toks = [t for t in generated_tokens if t < N_PITCHES]
    if not pitch_toks:
        return logits

    bias = torch.zeros(VOCAB_SIZE)

    for motif in motif_targets:
        motif_len = len(motif)   # number of intervals = number of notes - 1
        max_check = min(motif_len - 1, len(pitch_toks))

        for prefix_len in range(max_check, 0, -1):
            # We need prefix_len+1 recent pitches to compute prefix_len intervals
            recent = pitch_toks[-(prefix_len + 1):]
            if len(recent) < prefix_len + 1:
                continue

            recent_intervals = tuple(
                token_to_pitch(recent[i + 1]) - token_to_pitch(recent[i])
                for i in range(prefix_len)
            )

            if recent_intervals == motif[:prefix_len]:
                next_interval = motif[prefix_len]
                last_pitch    = token_to_pitch(pitch_toks[-1])
                target_pitch  = last_pitch + next_interval
                target_tok    = pitch_to_token(target_pitch)

                if 0 <= target_tok < N_PITCHES:
                    bias[target_tok] += MOTIF_BIAS_STRENGTH * strength

                break   # longest prefix only

    return logits + bias


# ---------------------------------------------------------------------------
# Bias helpers — voice leading
# ---------------------------------------------------------------------------

def _apply_repeat_penalty(
    logits:              torch.Tensor,
    last_pitch_token:    int,
    consecutive_repeats: int,
) -> torch.Tensor:
    """
    Apply a negative logit bias to the last generated pitch token, growing
    with the number of consecutive repeats.

    One immediate repeat (passing tone / ornament) is lightly penalised.
    Longer runs are progressively suppressed, preventing the Morse-code
    effect where the model locks onto a single pitch.

    penalty = REPEAT_PENALTY_BASE + (consecutive_repeats - 1) * REPEAT_PENALTY_SCALE
    """
    penalty = REPEAT_PENALTY_BASE + (consecutive_repeats - 1) * REPEAT_PENALTY_SCALE
    bias    = torch.zeros(VOCAB_SIZE)
    bias[last_pitch_token] = -penalty
    return logits + bias


def _apply_voice_leading_bias(
    logits:           torch.Tensor,
    arc_position:     float,
    chord_idx:        int,
    last_pitch_token: int,
) -> torch.Tensor:
    """
    Two complementary voice-leading nudges:

    1. Chord-tone targeting — grows linearly with arc_position so that by the
       end of the phrase the sax is being pushed toward root, 3rd, and 7th.
       No effect at the start (arc_position=0) so melodic freedom is preserved
       through the phrase body.

    2. Stepwise motion preference — a small constant bias toward pitches ±1 or
       ±2 semitones from the last generated pitch, encouraging smooth contrary
       motion rather than large leaps. Active throughout once a pitch exists.
    """
    from data.encoding import PITCH_MIN

    bias  = torch.zeros(VOCAB_SIZE)
    tones = chord_tones(chord_idx)

    # 1. Chord-tone endpoint targeting
    if tones and arc_position > 0:
        ct_strength = VOICE_LEADING_STRENGTH * arc_position
        for tok in range(N_PITCHES):
            if (PITCH_MIN + tok) % 12 in tones:
                bias[tok] += ct_strength

    # 2. Stepwise motion preference
    if last_pitch_token >= 0:
        for tok in range(N_PITCHES):
            interval = abs(tok - last_pitch_token)
            if interval <= 2:
                bias[tok] += STEPWISE_BIAS_STRENGTH * (1.0 - interval / 3.0)

    return logits + bias

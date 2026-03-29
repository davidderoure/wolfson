"""
Generates sax response phrases using the LSTM model.

Key additions over the basic generator:
  - Chord conditioning: passes chord_idx at each timestep
  - Contour steering: soft bias on pitch logits in the final portion of a phrase
    to guide the phrase toward a target contour (ascending / descending / neutral)
  - Swing/triplet bias: soft bias on duration logits toward the 12/8 triplet grid
    (1/3, 2/3, 4/3, 2 beats), enabling rhythmic contrast when the bass plays
    straight quarter notes — a 4-to-the-bar call gets a triplet-feel response
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import (
    phrase_to_tokens, phrase_to_chord_sequence, tokens_to_phrase,
    END_TOKEN, VOCAB_SIZE, DUR_OFFSET, N_PITCHES, N_DUR_BUCKETS,
    dur_to_token,
)
from data.chords import NC_INDEX, CHORD_VOCAB_SIZE
from generator.lstm_model import PhraseModel
from config import MAX_GENERATED_NOTES, GENERATION_TEMPERATURE, DEFAULT_INSTRUMENT

MODELS_DIR = Path(__file__).parent.parent / "models"

# Contour steering
CONTOUR_STEER_ONSET   = 0.6   # fraction of phrase before steering kicks in
CONTOUR_BIAS_STRENGTH = 1.5   # logit bias strength

# Swing/triplet bias
# Triplet grid in beat-relative durations: 1/3, 2/3, 1, 4/3, 2 beats
# These are the canonical durations of the 12/8 subdivision.
TRIPLET_DUR_BEATS = [1/3, 2/3, 1.0, 4/3, 2.0]
SWING_BIAS_STRENGTH = 1.2   # logit bias added to triplet-grid duration tokens


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
        seed_phrase:     list[dict],
        tempo_bpm:       float = 120.0,
        n_notes:         int   = None,
        temperature:     float = None,
        contour_target:  str   = "neutral",   # 'ascending', 'descending', 'neutral'
        chord_idx:       int   = NC_INDEX,
        swing_bias:      float = 0.0,         # 0 = no bias, 1 = full triplet-grid bias
    ) -> list[dict]:
        """
        Generate a sax phrase seeded by a bass phrase.

        seed_phrase:    list of note dicts (pitch, onset, offset, [beat_dur_sec], [chord_idx])
        tempo_bpm:      used only as fallback if notes lack beat_dur_sec
        contour_target: steer the ending of the phrase toward rising or falling pitch
        chord_idx:      current chord (NC_INDEX if unknown)
        swing_bias:     0–1, strength of triplet-grid duration bias.
                        Set to 1.0 when bass plays straight 4-to-bar to get a
                        swung/triplet response; 0.0 for no rhythmic steering.

        Returns: list of {pitch, duration_beats} dicts
        """
        n_notes     = n_notes     or MAX_GENERATED_NOTES
        temperature = temperature or GENERATION_TEMPERATURE

        seed_tokens = phrase_to_tokens(seed_phrase, tempo_bpm)
        seed_chords = phrase_to_chord_sequence(seed_phrase)

        # Override seed chord indices with the current chord if provided
        if chord_idx != NC_INDEX:
            seed_chords = [chord_idx] * len(seed_chords)

        tok_tensor   = torch.tensor([seed_tokens], dtype=torch.long)
        chord_tensor = torch.tensor([seed_chords], dtype=torch.long)

        generated_tokens = []
        hidden = None

        with torch.no_grad():
            # Prime hidden state with full seed
            logits, hidden = self.model(tok_tensor, chord_tensor, hidden)

            last_tok   = torch.tensor([[seed_tokens[-1]]], dtype=torch.long)
            last_chord = torch.tensor([[chord_idx]],       dtype=torch.long)
            expecting  = "pitch"
            note_count = 0

            for step in range(n_notes * 2):
                logits, hidden = self.model(last_tok, last_chord, hidden)
                tok_logits = logits[0, -1, :] / temperature

                # Enforce pitch/duration alternation
                tok_logits = tok_logits + _alternation_mask(expecting)

                # Contour steering: applied to pitch tokens in the final portion
                if expecting == "pitch" and note_count >= int(n_notes * CONTOUR_STEER_ONSET):
                    tok_logits = _apply_contour_bias(
                        tok_logits, generated_tokens, contour_target
                    )

                # Swing/triplet bias: applied to duration tokens throughout
                if expecting == "duration" and swing_bias > 0.0:
                    tok_logits = _apply_swing_bias(tok_logits, swing_bias)

                probs = F.softmax(tok_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()

                if token == END_TOKEN:
                    break

                generated_tokens.append(token)
                last_tok   = torch.tensor([[token]],      dtype=torch.long)
                last_chord = torch.tensor([[chord_idx]], dtype=torch.long)

                if expecting == "pitch":
                    note_count += 1
                    expecting = "duration"
                else:
                    expecting = "pitch"

        return tokens_to_phrase(generated_tokens)


# ---------------------------------------------------------------------------
# Precompute triplet-grid duration token set
# ---------------------------------------------------------------------------

def _build_triplet_token_set() -> set[int]:
    """
    Return the set of duration token indices that correspond to the triplet
    grid. We map each canonical triplet duration to its nearest bucket token,
    then also include the immediately adjacent buckets for robustness.
    """
    tokens = set()
    for dur_beats in TRIPLET_DUR_BEATS:
        t = dur_to_token(dur_beats)
        tokens.add(t)
        if t - 1 >= DUR_OFFSET:
            tokens.add(t - 1)
        if t + 1 < DUR_OFFSET + N_DUR_BUCKETS:
            tokens.add(t + 1)
    return tokens

_TRIPLET_TOKENS: set[int] = _build_triplet_token_set()


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _alternation_mask(expecting: str) -> torch.Tensor:
    """
    -inf mask for tokens of the wrong type.
    Enforces strict PITCH → DUR → PITCH → … alternation.
    """
    mask = torch.full((VOCAB_SIZE,), float("-inf"))
    if expecting == "pitch":
        mask[:N_PITCHES] = 0.0
        mask[END_TOKEN]  = 0.0
    else:
        mask[DUR_OFFSET:END_TOKEN] = 0.0
    return mask


def _apply_contour_bias(
    logits: torch.Tensor,
    generated_so_far: list[int],
    target: str,
) -> torch.Tensor:
    """
    Add a soft bias to pitch token logits to steer the phrase ending.

    'descending' → bias toward lower pitches (answer phrase / resolution)
    'ascending'  → bias toward higher pitches (question phrase / tension)
    'neutral'    → no bias

    The bias is proportional to distance from the phrase mean pitch,
    so it nudges gently rather than forcing a specific note.
    """
    if target == "neutral":
        return logits

    # Estimate mean pitch of generated phrase so far
    pitch_tokens = [t for t in generated_so_far if t < N_PITCHES]
    if not pitch_tokens:
        return logits

    mean_pitch_token = sum(pitch_tokens) / len(pitch_tokens)

    bias = torch.zeros(VOCAB_SIZE)
    for i in range(N_PITCHES):
        distance = i - mean_pitch_token
        if target == "descending":
            # Reward tokens below mean, penalise above
            bias[i] = -distance * (CONTOUR_BIAS_STRENGTH / N_PITCHES)
        else:  # ascending
            bias[i] = distance * (CONTOUR_BIAS_STRENGTH / N_PITCHES)

    return logits + bias


def _apply_swing_bias(logits: torch.Tensor, strength: float) -> torch.Tensor:
    """
    Add a positive bias to duration tokens that fall on the triplet (12/8) grid.

    strength 0.0 → no bias
    strength 1.0 → SWING_BIAS_STRENGTH added to triplet-grid tokens

    This nudges the model toward swung/triplet durations without forcing them —
    the LSTM's learned distribution still dominates.
    """
    bias = torch.zeros(VOCAB_SIZE)
    for t in _TRIPLET_TOKENS:
        bias[t] = SWING_BIAS_STRENGTH * strength
    return logits + bias

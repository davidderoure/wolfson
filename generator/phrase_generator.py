"""Generates sax response phrases using the LSTM model."""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import (
    phrase_to_tokens, tokens_to_phrase,
    pitch_to_token, dur_to_token,
    END_TOKEN, VOCAB_SIZE, DUR_OFFSET, N_PITCHES,
)
from generator.lstm_model import PhraseModel
from config import MAX_GENERATED_NOTES, GENERATION_TEMPERATURE, DEFAULT_INSTRUMENT

MODELS_DIR = Path(__file__).parent.parent / "models"


class PhraseGenerator:
    def __init__(self, instrument=None, model_path=None):
        instrument = instrument or DEFAULT_INSTRUMENT
        self.model = PhraseModel()

        if model_path is None:
            model_path = MODELS_DIR / f"{instrument}_best.pt"

        if Path(model_path).exists():
            checkpoint = torch.load(model_path, map_location="cpu")
            state = checkpoint["model"] if "model" in checkpoint else checkpoint
            self.model.load_state_dict(state)
            print(f"Loaded model from {model_path}")
        else:
            print(f"No model found at {model_path} — using untrained model (random output).")

        self.model.eval()

    def generate(self, seed_phrase: list[dict], tempo_bpm: float = 120.0,
                 n_notes: int = None, temperature: float = None) -> list[dict]:
        """
        Generate a response phrase seeded by a bass phrase.

        seed_phrase: list of note dicts with keys: pitch, onset, offset
        tempo_bpm:   used to convert note durations into beat-relative tokens
        Returns:     list of {pitch, duration_beats} dicts
        """
        n_notes    = n_notes    or MAX_GENERATED_NOTES
        temperature = temperature or GENERATION_TEMPERATURE

        seed_tokens = phrase_to_tokens(seed_phrase, tempo_bpm)
        input_tensor = torch.tensor([seed_tokens], dtype=torch.long)

        generated_tokens = []
        hidden = None

        with torch.no_grad():
            # Prime hidden state with seed phrase
            logits, hidden = self.model(input_tensor, hidden)

            # Generate up to n_notes pitch+duration pairs
            # We alternate: expect pitch token, then duration token
            last_token = torch.tensor([[seed_tokens[-1]]], dtype=torch.long)
            expecting = "pitch"   # next token type we want to generate

            for _ in range(n_notes * 2):   # *2 because each note is 2 tokens
                logits, hidden = self.model(last_token, hidden)
                token_logits = logits[0, -1, :] / temperature

                # Mask to only sample the expected token type
                mask = _make_mask(expecting)
                token_logits = token_logits + mask

                probs = F.softmax(token_logits, dim=-1)
                token = torch.multinomial(probs, 1).item()

                if token == END_TOKEN:
                    break

                generated_tokens.append(token)
                last_token = torch.tensor([[token]], dtype=torch.long)
                expecting = "duration" if expecting == "pitch" else "pitch"

        return tokens_to_phrase(generated_tokens)


def _make_mask(expecting: str) -> torch.Tensor:
    """
    Return a -inf mask for all token indices NOT of the expected type.
    This steers sampling to alternate strictly pitch → duration → pitch …
    """
    mask = torch.full((VOCAB_SIZE,), float("-inf"))
    if expecting == "pitch":
        mask[:N_PITCHES] = 0.0
        mask[END_TOKEN]  = 0.0   # allow early termination
    else:  # duration
        mask[DUR_OFFSET:DUR_OFFSET + (END_TOKEN - DUR_OFFSET)] = 0.0
    return mask

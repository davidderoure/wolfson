"""
LSTM model for jazz phrase generation with chord conditioning.

Vocabulary (output tokens) — defined in data/encoding.py:
  0  .. 49   pitch (MIDI 44-93)
  50 .. 81   duration bucket (log-scale, beat-relative)
  82         END

Sequence format: PITCH, DUR, PITCH, DUR, ..., END

Chord conditioning — defined in data/chords.py:
  0 .. 47    chord type (root × 4 + quality)
  48         NC (no chord / unknown)

Chord is an ADDITIONAL INPUT at each timestep, not a predicted output.
At inference time, pass chord_idx=NC_INDEX when no chord information is
available — the model degrades gracefully since NC was present in training.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import VOCAB_SIZE, N_PITCHES, N_DUR_BUCKETS, DUR_OFFSET, END_TOKEN
from data.chords import CHORD_VOCAB_SIZE, NC_INDEX
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS

PITCH_EMB_DIM = 32
DUR_EMB_DIM   = 16
CHORD_EMB_DIM = 16
LSTM_INPUT_DIM = PITCH_EMB_DIM + DUR_EMB_DIM + CHORD_EMB_DIM   # 64


class PhraseModel(nn.Module):
    """
    LSTM phrase model with separate embeddings for pitch, duration, and chord.

    Pitch and duration embeddings cover the note token types.
    Chord embedding is an additional conditioning signal concatenated to
    every timestep before the LSTM — giving the model harmonic context
    without adding chord tokens to the output vocabulary.
    """

    def __init__(self):
        super().__init__()

        # Note token embeddings (pitch and duration token types)
        self.pitch_embedding = nn.Embedding(N_PITCHES + 1, PITCH_EMB_DIM)   # +1 for END
        self.dur_embedding   = nn.Embedding(N_DUR_BUCKETS + 1, DUR_EMB_DIM)

        # Chord conditioning embedding
        self.chord_embedding = nn.Embedding(CHORD_VOCAB_SIZE, CHORD_EMB_DIM)

        self.lstm = nn.LSTM(
            input_size=LSTM_INPUT_DIM,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.3 if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, VOCAB_SIZE)

    def embed_notes(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, seq_len) mixed pitch/dur/end tokens
        Returns: (batch, seq_len, PITCH_EMB_DIM + DUR_EMB_DIM)
        """
        is_dur = (tokens >= DUR_OFFSET) & (tokens < END_TOKEN)
        is_end = (tokens == END_TOKEN)

        pitch_idx = tokens.clone()
        pitch_idx[is_dur] = 0
        pitch_idx[is_end] = N_PITCHES     # dedicated END embedding slot

        dur_idx = (tokens - DUR_OFFSET).clamp(0, N_DUR_BUCKETS)
        dur_idx[~is_dur] = N_DUR_BUCKETS  # zero-like embedding for non-dur positions

        p_emb = self.pitch_embedding(pitch_idx.clamp(0, N_PITCHES))
        d_emb = self.dur_embedding(dur_idx)

        # Zero out the wrong table at each position
        p_emb = p_emb * (~is_dur).unsqueeze(-1).float()
        d_emb = d_emb * is_dur.unsqueeze(-1).float()

        return torch.cat([p_emb, d_emb], dim=-1)

    def forward(
        self,
        tokens: torch.Tensor,           # (batch, seq_len)
        chords: torch.Tensor,           # (batch, seq_len) chord indices
        hidden=None,
    ):
        """
        Returns: logits (batch, seq_len, VOCAB_SIZE), new hidden state
        """
        note_emb  = self.embed_notes(tokens)                          # (B, T, 48)
        chord_emb = self.chord_embedding(chords.clamp(0, CHORD_VOCAB_SIZE - 1))  # (B, T, 16)
        lstm_in   = torch.cat([note_emb, chord_emb], dim=-1)         # (B, T, 64)

        out, hidden = self.lstm(lstm_in, hidden)
        logits = self.fc(out)
        return logits, hidden

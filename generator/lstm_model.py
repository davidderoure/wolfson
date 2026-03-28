"""
LSTM model for sax (and future instrument) phrase generation.

Vocabulary is defined in data/encoding.py:
  tokens 0-49:   pitch (MIDI 44-93)
  tokens 50-81:  duration bucket (log-scale, beat-relative)
  token  82:     END

Sequence format: PITCH, DUR, PITCH, DUR, ..., END
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data.encoding import VOCAB_SIZE, N_PITCHES, N_DUR_BUCKETS, DUR_OFFSET, END_TOKEN
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS


class PhraseModel(nn.Module):
    """
    Single LSTM that generates interleaved pitch+duration token sequences.

    Separate embeddings for pitch tokens and duration tokens are concatenated
    before being fed to the LSTM, giving the model a structured view of the
    two token types while sharing a single recurrent state.
    """

    PITCH_EMB_DIM = 32
    DUR_EMB_DIM   = 16
    EMB_DIM       = PITCH_EMB_DIM + DUR_EMB_DIM   # 48 total

    def __init__(self):
        super().__init__()
        # Separate embedding tables for pitch and duration tokens
        self.pitch_embedding = nn.Embedding(N_PITCHES + 1, self.PITCH_EMB_DIM)  # +1 for END
        self.dur_embedding   = nn.Embedding(N_DUR_BUCKETS + 1, self.DUR_EMB_DIM)

        self.lstm = nn.LSTM(
            input_size=self.EMB_DIM,
            hidden_size=LSTM_HIDDEN_SIZE,
            num_layers=LSTM_NUM_LAYERS,
            batch_first=True,
            dropout=0.3 if LSTM_NUM_LAYERS > 1 else 0.0,
        )
        self.fc = nn.Linear(LSTM_HIDDEN_SIZE, VOCAB_SIZE)

    def embed(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (batch, seq_len) of mixed pitch/dur/end tokens
        Returns: (batch, seq_len, EMB_DIM)
        """
        # Pitch tokens: 0 .. N_PITCHES-1  → pitch embedding index
        # Dur tokens:   DUR_OFFSET ..     → dur embedding index (subtract offset)
        # END token:    treated as pitch index N_PITCHES (out-of-range → zero-like)
        is_dur = (tokens >= DUR_OFFSET) & (tokens < END_TOKEN)
        is_end = (tokens == END_TOKEN)

        pitch_idx = tokens.clone()
        pitch_idx[is_dur] = 0          # masked out below
        pitch_idx[is_end] = N_PITCHES  # special end embedding

        dur_idx = tokens.clone() - DUR_OFFSET
        dur_idx[~is_dur] = N_DUR_BUCKETS   # zero embedding for non-dur positions

        p_emb = self.pitch_embedding(pitch_idx.clamp(0, N_PITCHES))
        d_emb = self.dur_embedding(dur_idx.clamp(0, N_DUR_BUCKETS))

        # Zero out the wrong embedding at each position
        p_emb = p_emb * (~is_dur).unsqueeze(-1).float()
        d_emb = d_emb * is_dur.unsqueeze(-1).float()

        return torch.cat([p_emb, d_emb], dim=-1)

    def forward(self, x: torch.Tensor, hidden=None):
        """
        x:      (batch, seq_len) token indices
        hidden: LSTM hidden state tuple or None
        Returns: logits (batch, seq_len, VOCAB_SIZE), new hidden state
        """
        emb = self.embed(x)
        out, hidden = self.lstm(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

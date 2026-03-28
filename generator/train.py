"""
Training script for the Wolfson phrase model.

Usage:
    python generator/train.py --instrument sax
    python generator/train.py --instrument sax --epochs 100 --batch-size 64
    python generator/train.py --instrument sax --resume models/sax_latest.pt

Reads:
    data/processed/sax_sequences.npy    token sequences
    data/processed/sax_chords.npy       chord index sequences (same length)
Writes:
    models/sax_best.pt
    models/sax_latest.pt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import VOCAB_SIZE, END_TOKEN
from data.chords import NC_INDEX, CHORD_VOCAB_SIZE
from generator.lstm_model import PhraseModel
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhraseDataset(Dataset):
    """
    Each item: (input_tokens, input_chords, target_tokens)

    target_tokens is input_tokens shifted left by one (next-token prediction).
    Chord sequence aligns with input_tokens (same length, shifted along with it).
    """

    def __init__(self, sequences: list, chord_seqs: list, max_len: int = 128):
        self.items = []
        for seq, chords in zip(sequences, chord_seqs):
            if len(seq) > max_len:
                seq    = seq[:max_len]
                chords = chords[:max_len]
            if len(seq) < 3:
                continue
            t = torch.tensor(seq,    dtype=torch.long)
            c = torch.tensor(chords, dtype=torch.long)
            # Align: input uses positions 0..n-2, target is 1..n-1
            # Chord conditioning: use chord at input position (predicting the next token
            # given current token + current chord)
            self.items.append((t[:-1], c[:-1], t[1:]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    inputs, chords, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)

    def pad(seqs, pad_val):
        out = torch.full((len(seqs), max_len), pad_val, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return pad(inputs, END_TOKEN), pad(chords, NC_INDEX), pad(targets, END_TOKEN)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seq_path   = PROCESSED_DIR / f"{args.instrument}_sequences.npy"
    chord_path = PROCESSED_DIR / f"{args.instrument}_chords.npy"

    for p in (seq_path, chord_path):
        if not p.exists():
            print(f"ERROR: {p} not found. Run data/prepare.py --instrument {args.instrument} first.")
            sys.exit(1)

    print(f"Loading from {PROCESSED_DIR} ...")
    sequences  = list(np.load(seq_path,   allow_pickle=True))
    chord_seqs = list(np.load(chord_path, allow_pickle=True))
    print(f"  {len(sequences)} phrases loaded.")

    dataset    = PhraseDataset(sequences, chord_seqs, max_len=args.max_len)
    print(f"  {len(dataset)} training items after filtering.")

    val_size   = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Training on {device}.")

    model = PhraseModel().to(device)

    start_epoch   = 0
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        start_epoch   = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", best_val_loss)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5
    )
    criterion = nn.CrossEntropyLoss(ignore_index=END_TOKEN)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {n_params:,}")
    print(f"Vocab: {VOCAB_SIZE} tokens  Chords: {CHORD_VOCAB_SIZE}  "
          f"Hidden: {LSTM_HIDDEN_SIZE}  Layers: {LSTM_NUM_LAYERS}\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for inp_tokens, inp_chords, targets in train_loader:
            inp_tokens = inp_tokens.to(device)
            inp_chords = inp_chords.to(device)
            targets    = targets.to(device)

            optimiser.zero_grad()
            logits, _ = model(inp_tokens, inp_chords)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inp_tokens, inp_chords, targets in val_loader:
                inp_tokens = inp_tokens.to(device)
                inp_chords = inp_chords.to(device)
                targets    = targets.to(device)
                logits, _  = model(inp_tokens, inp_chords)
                val_loss  += criterion(
                    logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
                ).item()
        val_loss /= len(val_loader)

        prev_lr = optimiser.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr  = optimiser.param_groups[0]["lr"]
        elapsed = time.time() - t0
        lr_note = f"  lr→{new_lr:.2e}" if new_lr != prev_lr else ""
        print(f"Epoch {epoch+1:4d}  train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.1f}s{lr_note}")

        ckpt = {
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "instrument": args.instrument,
        }
        torch.save(ckpt, MODELS_DIR / f"{args.instrument}_latest.pt")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt["best_val_loss"] = best_val_loss
            torch.save(ckpt, MODELS_DIR / f"{args.instrument}_best.pt")
            print(f"          *** new best ({best_val_loss:.4f}) ***")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Wolfson phrase model")
    parser.add_argument("--instrument", default="sax")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max-len",    type=int,   default=128)
    parser.add_argument("--resume",     default=None)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

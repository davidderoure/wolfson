"""
Training script for the Wolfson phrase model.

Usage:
    python generator/train.py --instrument sax
    python generator/train.py --instrument sax --epochs 100 --batch-size 64
    python generator/train.py --instrument sax --resume models/sax_latest.pt

Reads:
    data/processed/sax_sequences.npy
Writes:
    models/sax_best.pt
    models/sax_latest.pt
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import VOCAB_SIZE, END_TOKEN
from generator.lstm_model import PhraseModel
from config import LSTM_HIDDEN_SIZE, LSTM_NUM_LAYERS

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
MODELS_DIR    = Path(__file__).parent.parent / "models"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PhraseDataset(Dataset):
    """
    Each item is a (input_tokens, target_tokens) pair where target is
    input shifted left by one — standard next-token prediction.
    """

    def __init__(self, sequences: list[np.ndarray], max_len: int = 128):
        self.items = []
        for seq in sequences:
            # Truncate very long sequences
            if len(seq) > max_len:
                seq = seq[:max_len]
            if len(seq) < 3:
                continue
            t = torch.tensor(seq, dtype=torch.long)
            self.items.append((t[:-1], t[1:]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate_fn(batch):
    """Pad sequences in a batch to the same length."""
    inputs, targets = zip(*batch)
    max_len = max(x.size(0) for x in inputs)
    pad = END_TOKEN

    def pad_seq(seqs):
        out = torch.full((len(seqs), max_len), pad, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, :s.size(0)] = s
        return out

    return pad_seq(inputs), pad_seq(targets)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    seq_path = PROCESSED_DIR / f"{args.instrument}_sequences.npy"
    if not seq_path.exists():
        print(f"ERROR: {seq_path} not found. Run data/prepare.py --instrument {args.instrument} first.")
        sys.exit(1)

    print(f"Loading sequences from {seq_path} ...")
    sequences = list(np.load(seq_path, allow_pickle=True))
    print(f"  {len(sequences)} phrases loaded.")

    dataset = PhraseDataset(sequences, max_len=args.max_len)
    print(f"  {len(dataset)} training items after filtering.")

    val_size  = max(1, int(len(dataset) * 0.1))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size],
                                      generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Training on {device}.")

    model = PhraseModel().to(device)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resumed from epoch {start_epoch}, best val loss {best_val_loss:.4f}")

    optimiser = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, patience=5, factor=0.5, verbose=True
    )
    criterion = nn.CrossEntropyLoss(ignore_index=END_TOKEN)

    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Vocab size: {VOCAB_SIZE}   Hidden: {LSTM_HIDDEN_SIZE}   Layers: {LSTM_NUM_LAYERS}\n")

    for epoch in range(start_epoch, start_epoch + args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        t0 = time.time()

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimiser.zero_grad()
            logits, _ = model(inputs)
            # logits: (batch, seq_len, vocab); targets: (batch, seq_len)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                logits, _ = model(inputs)
                val_loss += criterion(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)).item()
        val_loss /= len(val_loader)

        scheduler.step(val_loss)

        elapsed = time.time() - t0
        print(f"Epoch {epoch+1:4d}  train={train_loss:.4f}  val={val_loss:.4f}  {elapsed:.1f}s")

        # Save latest
        latest_path = MODELS_DIR / f"{args.instrument}_latest.pt"
        torch.save({
            "model": model.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss,
            "instrument": args.instrument,
        }, latest_path)

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = MODELS_DIR / f"{args.instrument}_best.pt"
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "instrument": args.instrument,
            }, best_path)
            print(f"          *** new best model saved ({best_val_loss:.4f}) ***")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train Wolfson phrase model")
    parser.add_argument("--instrument", default="sax", help="Instrument family")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--max-len",    type=int,   default=128,
                        help="Max token sequence length (longer phrases are truncated)")
    parser.add_argument("--resume",     type=str,   default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

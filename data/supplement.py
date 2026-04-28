"""
Supplement Wolfson training data with additional MIDI melody files.

Processes each file with the same phrase segmentation and token encoding as
prepare.py, then appends the new phrases to the existing processed data in
data/processed/.  Chord annotations are set to NC (no chord) throughout —
the model learns the melodic patterns chord-agnostically.

Backup files (*.bak) are written before any modification so the original
training data can be restored if needed.

Usage:
    python data/supplement.py Summertime.mid AutumnLeaves.mid ...

Logic export checklist:
    • Solo the melody track, delete any chord/pad notes — monophonic only.
    • Name the Logic track (e.g. "Tenor Sax") — recorded in metadata.
    • Set project tempo to the intended feel before export.
    • File → Export → Selection as MIDI File (Type 1 is fine).

To restore the original training data:
    cp data/processed/sax_sequences.npy.bak data/processed/sax_sequences.npy
    cp data/processed/sax_chords.npy.bak    data/processed/sax_chords.npy
    cp data/processed/sax_meta.json.bak     data/processed/sax_meta.json
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pretty_midi

# Allow running from repo root or from data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.encoding import phrase_to_tokens, phrase_to_chord_sequence
from data.prepare import segment_phrases

PROCESSED_DIR = Path(__file__).parent / "processed"
_FILES = ("sax_sequences.npy", "sax_chords.npy", "sax_meta.json")


# ---------------------------------------------------------------------------
# Load / save processed data
# ---------------------------------------------------------------------------

def _load_existing() -> tuple[list, list, list]:
    for name in _FILES:
        if not (PROCESSED_DIR / name).exists():
            print(f"ERROR: {PROCESSED_DIR / name} not found.")
            print("Run  python data/prepare.py --instrument sax  first.")
            sys.exit(1)

    seqs   = list(np.load(PROCESSED_DIR / "sax_sequences.npy", allow_pickle=True))
    chords = list(np.load(PROCESSED_DIR / "sax_chords.npy",    allow_pickle=True))
    with open(PROCESSED_DIR / "sax_meta.json") as f:
        meta = json.load(f)

    return seqs, chords, meta


def _backup():
    for name in _FILES:
        src = PROCESSED_DIR / name
        dst = src.with_suffix(src.suffix + ".bak")
        shutil.copy2(src, dst)
        print(f"  Backed up  {name}  →  {dst.name}")


def _save(seqs: list, chords: list, meta: list) -> None:
    np.save(PROCESSED_DIR / "sax_sequences.npy",
            np.array(seqs,   dtype=object), allow_pickle=True)
    np.save(PROCESSED_DIR / "sax_chords.npy",
            np.array(chords, dtype=object), allow_pickle=True)
    with open(PROCESSED_DIR / "sax_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# MIDI processing
# ---------------------------------------------------------------------------

def _process_midi(path: Path) -> tuple[list, list, list]:
    """
    Load a MIDI file and return (sequences, chord_seqs, meta_entries).

    All non-drum tracks are merged into a single note stream.  Phrases are
    split by the standard 0.5 s gap and encoded with NC chord indices.
    """
    try:
        mid = pretty_midi.PrettyMIDI(str(path))
    except Exception as exc:
        print(f"  Warning: could not parse {path.name}: {exc}")
        return [], [], []

    tempo_bpm  = float(mid.estimate_tempo() or 120.0)
    track_name = ""

    notes: list[dict] = []
    for inst in mid.instruments:
        if inst.is_drum:
            continue
        if not track_name and inst.name:
            track_name = inst.name
        for note in inst.notes:
            notes.append({
                "pitch":  note.pitch,
                "onset":  note.start,
                "offset": note.end,
            })

    if not notes:
        print(f"  No notes found in {path.name}.")
        return [], [], []

    notes.sort(key=lambda n: n["onset"])
    phrases = segment_phrases(notes)

    if not phrases:
        total = len(notes)
        print(f"  {total} note(s) found but no phrases met the "
              f"length filter (3–64 notes, ≥0.5 s gaps).")
        print(f"  If the melody is one continuous line with no rests, "
              f"add a short rest (≥0.5 s) between phrases in Logic.")
        return [], [], []

    seqs, chord_seqs, metas = [], [], []
    for phrase in phrases:
        seqs.append(       np.array(phrase_to_tokens(phrase, tempo_bpm),       dtype=np.int16))
        chord_seqs.append( np.array(phrase_to_chord_sequence(phrase),          dtype=np.int8))
        metas.append({
            "melid":      path.stem,
            "title":      path.stem,
            "performer":  track_name or path.stem,
            "instrument": "melody",
            "tempo_bpm":  tempo_bpm,
            "n_notes":    len(phrase),
        })

    return seqs, chord_seqs, metas


# ---------------------------------------------------------------------------
# list-supplements helper
# ---------------------------------------------------------------------------

def _list_supplements(meta: list) -> None:
    entries = [m for m in meta if m.get("instrument") == "melody"]
    if not entries:
        print("No supplementary melody files in current training data.")
        return
    titles = {}
    for m in entries:
        titles.setdefault(m["title"], []).append(m["n_notes"])
    print(f"{'Title':<40}  {'Phrases':>7}  {'Notes (per phrase)':}")
    print("-" * 70)
    for title, note_counts in sorted(titles.items()):
        print(f"{title:<40}  {len(note_counts):>7}  "
              f"{', '.join(str(n) for n in note_counts)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplement Wolfson training data with MIDI melody files.")
    parser.add_argument(
        "midi_files", nargs="*", metavar="FILE",
        help="MIDI files to add (omit to use --list)")
    parser.add_argument(
        "--list", action="store_true",
        help="List melody files already added to the training data")
    args = parser.parse_args()

    print(f"Loading existing training data from {PROCESSED_DIR} ...")
    seqs, chords, meta = _load_existing()
    n_existing = len(seqs)
    print(f"  {n_existing} existing phrases.")

    if args.list:
        _list_supplements(meta)
        return

    if not args.midi_files:
        parser.print_help()
        sys.exit(0)

    paths = [Path(p) for p in args.midi_files]
    missing = [p for p in paths if not p.exists()]
    if missing:
        for p in missing:
            print(f"ERROR: file not found: {p}")
        sys.exit(1)

    # Collect new phrases
    new_seqs, new_chords, new_meta = [], [], []
    for path in paths:
        print(f"\nProcessing  {path.name} ...")
        s, c, m = _process_midi(path)
        if not s:
            continue
        new_seqs.extend(s)
        new_chords.extend(c)
        new_meta.extend(m)
        print(f"  {len(s)} phrase(s) extracted  "
              f"(notes per phrase: {', '.join(str(e['n_notes']) for e in m)})  "
              f"tempo: {m[0]['tempo_bpm']:.0f} BPM  "
              f"track: \"{m[0]['performer']}\"")

    if not new_seqs:
        print("\nNo new phrases extracted — nothing to add.")
        sys.exit(0)

    # Back up then save
    print(f"\nBacking up existing data ...")
    _backup()

    _save(seqs + new_seqs, chords + new_chords, meta + new_meta)

    print(f"\nDone.")
    print(f"  Phrases added:   {len(new_seqs)}")
    print(f"  Total phrases:   {n_existing + len(new_seqs)}")
    print(f"\nNext step:  retrain the model")
    print(f"  python generator/train.py   (or open wolfson_train.ipynb)")
    print(f"\nTo list what's been added:")
    print(f"  python data/supplement.py --list")
    print(f"\nTo restore original data:")
    for name in _FILES:
        p   = PROCESSED_DIR / name
        bak = p.with_suffix(p.suffix + ".bak")
        print(f"  cp {bak}  {p}")


if __name__ == "__main__":
    main()

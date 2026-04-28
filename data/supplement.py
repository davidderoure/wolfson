"""
Supplement Wolfson training data with additional MIDI melody files.

Processes each file with the same phrase segmentation and token encoding as
prepare.py, then appends the new phrases to the existing processed data in
data/processed/.  Chord annotations are set to NC (no chord) throughout —
the model learns the melodic patterns chord-agnostically.

Backup files (*.bak) are written before any modification so the original
training data can be restored if needed.

Usage:
    # Basic — add phrases as-is (very small impact on a ~5000-phrase corpus)
    python data/supplement.py Summertime.mid FlyMeToTheMoon.mid

    # Transpose to all 12 keys — multiplies phrase count × 12 (~5% of corpus)
    python data/supplement.py --transpose Summertime.mid FlyMeToTheMoon.mid

    # Repeat each phrase N times — recommended when NOT transposing
    python data/supplement.py --repeat 20 Summertime.mid FlyMeToTheMoon.mid

    # Both — each phrase × 12 keys × N repeats
    python data/supplement.py --transpose --repeat 3 Summertime.mid ...

    # Show what has already been added
    python data/supplement.py --list

Recommended starting point:
    --transpose --repeat 3   →  ~360 phrases from 2 heads  (~7% of corpus)

Logic export checklist:
    • Solo the melody track, delete any chord/pad notes — monophonic only.
    • Name the Logic track (e.g. "Tenor Sax") — recorded in metadata.
    • Set project tempo to the intended feel before export.
    • File → Export → Selection as MIDI File (Type 1 is fine).
    • Add a short rest (≥ 0.5 s) between natural phrase boundaries so the
      segmenter splits the melody into individual phrases rather than one block.

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
_FILES        = ("sax_sequences.npy", "sax_chords.npy", "sax_meta.json")

# All 12 semitone shifts: 0 = original key, 1–11 = upward transpositions.
# Pitches outside the encoding range (Ab2–A6) are clipped by pitch_to_token,
# so no phrases are discarded — out-of-range notes just hit the boundary.
_ALL_SHIFTS = list(range(12))


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


def _backup() -> None:
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
# Transposition
# ---------------------------------------------------------------------------

def _transpose_phrase(phrase: list[dict], semitones: int) -> list[dict]:
    """Return a copy of phrase with every pitch shifted by semitones."""
    return [{**note, "pitch": note["pitch"] + semitones} for note in phrase]


# ---------------------------------------------------------------------------
# MIDI processing
# ---------------------------------------------------------------------------

def _process_midi(
    path:      Path,
    transpose: bool = False,
    repeat:    int  = 1,
) -> tuple[list, list, list]:
    """
    Load a MIDI file and return (sequences, chord_seqs, meta_entries).

    All non-drum tracks are merged into a single note stream and split into
    phrases at silence gaps ≥ 0.5 s.  For each base phrase:
      • If --transpose: generates all 12 semitone transpositions.
      • Each (possibly transposed) phrase is repeated `repeat` times.

    Chord annotations are set to NC throughout (no chord data in the file).
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
    base_phrases = segment_phrases(notes)

    if not base_phrases:
        print(f"  {len(notes)} note(s) found but no phrases met the length "
              f"filter (3–64 notes, ≥ 0.5 s gaps).")
        print(f"  Add short rests between phrases in Logic and re-export.")
        return [], [], []

    shifts = _ALL_SHIFTS if transpose else [0]

    seqs, chord_seqs, metas = [], [], []
    for phrase in base_phrases:
        for shift in shifts:
            p = _transpose_phrase(phrase, shift) if shift else phrase
            tokens     = phrase_to_tokens(p, tempo_bpm)
            chord_seq  = phrase_to_chord_sequence(p)   # all NC_INDEX
            for _ in range(repeat):
                seqs.append(      np.array(tokens,    dtype=np.int16))
                chord_seqs.append(np.array(chord_seq, dtype=np.int8))
                metas.append({
                    "melid":        path.stem,
                    "title":        path.stem,
                    "performer":    track_name or path.stem,
                    "instrument":   "melody",
                    "tempo_bpm":    tempo_bpm,
                    "n_notes":      len(phrase),
                    "transposition": shift,
                    "repeat":        repeat,
                })

    return seqs, chord_seqs, metas


# ---------------------------------------------------------------------------
# --list helper
# ---------------------------------------------------------------------------

def _list_supplements(meta: list) -> None:
    entries = [m for m in meta if m.get("instrument") == "melody"]
    if not entries:
        print("No supplementary melody files in current training data.")
        return

    # Group by title
    by_title: dict[str, list[dict]] = {}
    for m in entries:
        by_title.setdefault(m["title"], []).append(m)

    print(f"\n{'Title':<36}  {'Phrases':>7}  {'Base':>5}  "
          f"{'Keys':>5}  {'Repeat':>6}")
    print("─" * 66)
    for title, rows in sorted(by_title.items()):
        n_phrases    = len(rows)
        base_phrases = len({(m["n_notes"], m.get("transposition", 0) == 0)
                            for m in rows if m.get("transposition", 0) == 0})
        # Infer from metadata
        shifts  = len({m.get("transposition", 0) for m in rows})
        repeats = rows[0].get("repeat", 1)
        print(f"{title:<36}  {n_phrases:>7}  {base_phrases:>5}  "
              f"{shifts:>5}  {repeats:>6}")

    print(f"\nTotal supplementary phrases: {len(entries)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Supplement Wolfson training data with MIDI melody files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python data/supplement.py --transpose --repeat 3 "
            "Summertime.mid FlyMeToTheMoon.mid\n"
            "      → 2 heads × 12 keys × 3 repeats = ~360 phrases\n\n"
            "  python data/supplement.py --list\n"
            "      → show what has already been added"
        ),
    )
    parser.add_argument(
        "midi_files", nargs="*", metavar="FILE",
        help="MIDI files to add")
    parser.add_argument(
        "--transpose", action="store_true",
        help="Generate all 12 semitone transpositions of each phrase "
             "(multiplies phrase count × 12)")
    parser.add_argument(
        "--repeat", type=int, default=1, metavar="N",
        help="Repeat each phrase N times in the training corpus (default: 1). "
             "Applied per transposition when combined with --transpose.")
    parser.add_argument(
        "--list", action="store_true",
        help="List melody files already added to the training data")
    args = parser.parse_args()

    if args.repeat < 1:
        parser.error("--repeat must be ≥ 1")

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

    n_keys = 12 if args.transpose else 1
    print(f"\nSettings:  transpose={'all 12 keys' if args.transpose else 'off'}  "
          f"repeat={args.repeat}  "
          f"(each base phrase → {n_keys * args.repeat} training entries)")

    # Collect new phrases
    new_seqs, new_chords, new_meta = [], [], []
    for path in paths:
        print(f"\nProcessing  {path.name} ...")
        s, c, m = _process_midi(path, transpose=args.transpose, repeat=args.repeat)
        if not s:
            continue

        base = len({(e["n_notes"], e["transposition"]) for e in m
                    if e["transposition"] == 0})
        print(f"  Base phrases: {base}  →  {len(s)} training entries  "
              f"({n_keys} key{'s' if n_keys > 1 else ''} × {args.repeat} repeat{'s' if args.repeat > 1 else ''})  "
              f"tempo: {m[0]['tempo_bpm']:.0f} BPM  "
              f"track: \"{m[0]['performer']}\"")

        new_seqs.extend(s)
        new_chords.extend(c)
        new_meta.extend(m)

    if not new_seqs:
        print("\nNo new phrases extracted — nothing to add.")
        sys.exit(0)

    pct = 100 * len(new_seqs) / (n_existing + len(new_seqs))
    print(f"\nBacking up existing data ...")
    _backup()

    _save(seqs + new_seqs, chords + new_chords, meta + new_meta)

    print(f"\nDone.")
    print(f"  Phrases added:   {len(new_seqs)}  "
          f"({pct:.1f}% of new total)")
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

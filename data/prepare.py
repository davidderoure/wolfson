"""
Wolfson data preparation script.

Extracts solo phrases from the Weimar Jazz Database (WJD), encodes them as
pitch+duration token sequences, and saves training data.

Two data sources are supported — use whichever you have downloaded:

  A) SQLite database + MIDI files (preferred, more metadata)
       data/raw/wjazzd.db            SQLite database
       data/raw/midi_unquant/        Unquantised MIDI files
                                     (extract RELEASE2.0_mid_unquant.zip here)

  B) MIDI files only (--midi-only)
       data/raw/midi_unquant/        As above; instrument inferred from filename

Usage:
    # Inspect database schema and available instruments (requires wjazzd.db):
    python data/prepare.py --inspect

    # Extract saxophone phrases using database + MIDI:
    python data/prepare.py --instrument sax

    # Extract using MIDI files only (no database required):
    python data/prepare.py --instrument sax --midi-only
"""

import argparse
import bisect
import json
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pretty_midi

# Allow running from repo root or from data/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.instruments import codes_for, PITCH_RANGES, family_for_code
from data.encoding import phrase_to_tokens, phrase_to_chord_sequence, VOCAB_SIZE, END_TOKEN
from data.chords import parse_chord, NC_INDEX

RAW_DIR       = Path(__file__).parent / "raw"
PROCESSED_DIR = Path(__file__).parent / "processed"
WJD_DB        = RAW_DIR / "wjazzd.db"

# The unquantised MIDI zip: RELEASE2.0_mid_unquant.zip
# Extract this into data/raw/midi_unquant/ before running.
# The SQLite database (wjazzd.db) is a separate download from the same page.
WJD_MIDI_DIR  = RAW_DIR / "midi_unquant"

# A gap of this many seconds between notes ends a phrase
PHRASE_GAP_SEC = 0.5
MIN_PHRASE_NOTES = 3
MAX_PHRASE_NOTES = 64   # discard unusually long runs (likely data errors)


# ---------------------------------------------------------------------------
# Database inspection
# ---------------------------------------------------------------------------

def inspect(db: sqlite3.Connection) -> None:
    cur = db.cursor()

    print("=== Tables ===")
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = [r[0] for r in cur.fetchall()]
    for t in tables:
        print(f"  {t}")

    for t in tables:
        print(f"\n=== {t} columns ===")
        cur.execute(f"PRAGMA table_info({t})")
        for row in cur.fetchall():
            print(f"  {row[1]:30s}  {row[2]}")

    print("\n=== Instrument distribution ===")
    try:
        cur.execute("""
            SELECT instrument, COUNT(*) AS n
            FROM solo_info
            GROUP BY instrument
            ORDER BY n DESC
        """)
        for code, n in cur.fetchall():
            family = family_for_code(code) or "—"
            print(f"  {code:6s}  {n:4d} solos   (family: {family})")
    except sqlite3.OperationalError as e:
        print(f"  Could not query solo_info: {e}")


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

def load_solos(db: sqlite3.Connection, instrument_codes: list[str]) -> list[dict]:
    """
    Return a list of solos, each a dict with metadata and a sorted list of notes.
    Notes have: pitch (int), onset (float, sec), offset (float, sec).

    Chord assignment uses two bulk queries per solo (notes + beats) and bisect
    rather than a correlated subquery per note, which is ~100x faster on the
    full sax dataset (~106k notes).
    """
    cur = db.cursor()
    placeholders = ",".join("?" * len(instrument_codes))

    # Fetch solo metadata — avgtempo is in BPM
    cur.execute(f"""
        SELECT melid, title, performer, instrument, avgtempo
        FROM solo_info
        WHERE instrument IN ({placeholders})
        ORDER BY melid
    """, instrument_codes)
    rows = cur.fetchall()

    # Bulk-fetch all beats (chord changes) for matching solos in one query,
    # keyed by melid so we can look them up per solo without extra round-trips.
    melids = [r[0] for r in rows]
    mel_placeholders = ",".join("?" * len(melids))
    cur.execute(f"""
        SELECT melid, onset, chord
        FROM beats
        WHERE melid IN ({mel_placeholders})
        ORDER BY melid, onset
    """, melids)
    beats_by_melid: dict[int, tuple[list, list]] = {}
    for melid, onset, chord in cur.fetchall():
        if melid not in beats_by_melid:
            beats_by_melid[melid] = ([], [])
        beats_by_melid[melid][0].append(float(onset))
        beats_by_melid[melid][1].append(chord)

    solos = []
    for melid, title, performer, instrument, avgtempo in rows:
        tempo_bpm = float(avgtempo) if avgtempo else 120.0

        cur.execute("""
            SELECT pitch, onset, duration, beatdur
            FROM melody
            WHERE melid = ?
            ORDER BY onset
        """, (melid,))

        beat_onsets, beat_chords = beats_by_melid.get(melid, ([], []))

        notes = []
        for pitch, onset, duration, beatdur in cur.fetchall():
            if pitch is None or onset is None or duration is None:
                continue
            beat_dur_sec = float(beatdur) if beatdur else (60.0 / tempo_bpm)

            # Find the most recent chord change at or before this note's onset
            # using bisect — O(log n) per note instead of a subquery.
            chord_str = None
            if beat_onsets:
                idx = bisect.bisect_right(beat_onsets, float(onset)) - 1
                if idx >= 0:
                    chord_str = beat_chords[idx]

            notes.append({
                "pitch":        int(round(float(pitch))),
                "onset":        float(onset),
                "offset":       float(onset) + float(duration),
                "beat_dur_sec": beat_dur_sec,
                "chord_idx":    parse_chord(chord_str) if chord_str else NC_INDEX,
            })

        solos.append({
            "melid":      melid,
            "title":      title,
            "performer":  performer,
            "instrument": instrument,
            "tempo_bpm":  tempo_bpm,
            "notes":      notes,
        })

    return solos



# ---------------------------------------------------------------------------
# Phrase segmentation
# ---------------------------------------------------------------------------

def segment_phrases(notes: list[dict]) -> list[list[dict]]:
    """
    Split a note list into phrases by silence gaps >= PHRASE_GAP_SEC.
    Returns only phrases within [MIN_PHRASE_NOTES, MAX_PHRASE_NOTES].
    """
    if not notes:
        return []

    phrases = []
    current = [notes[0]]

    for note in notes[1:]:
        gap = note["onset"] - current[-1]["offset"]
        if gap >= PHRASE_GAP_SEC:
            if MIN_PHRASE_NOTES <= len(current) <= MAX_PHRASE_NOTES:
                phrases.append(current)
            current = [note]
        else:
            current.append(note)

    if MIN_PHRASE_NOTES <= len(current) <= MAX_PHRASE_NOTES:
        phrases.append(current)

    return phrases


# ---------------------------------------------------------------------------
# MIDI-only extraction (from RELEASE2.0_mid_unquant.zip)
# ---------------------------------------------------------------------------

# WJD MIDI filenames encode performer and instrument, e.g.:
#   BirdParker_Confirmation-1_FINAL.mid   → performer BirdParker, inferred as 'as'
# The instrument code is not always in the filename; we use a lookup table
# from known performers where possible, and fall back to track name heuristics.

_PERFORMER_INSTRUMENT: dict[str, str] = {
    # Alto sax
    "BirdParker":   "as",
    "LeeKonitz":    "as",
    "PaulDesmond":  "as",
    "BennyCarterAS":"as",
    # Tenor sax
    "ColtraneTenor":"ts",
    "StanGetz":     "ts",
    "ZootSims":     "ts",
    "WarneMarch":   "ts",  # Marsh
    "LesterYoung":  "ts",
    "DonByas":      "ts",
    "ColemanHawkins":"ts",
    "BenWebster":   "ts",
    # Trumpet
    "MilesDavis":   "tp",
    "ChetBaker":    "tp",
    "CliffordBrown":"tp",
    # Trombone
    "JJJohnson":    "tb",
}


def _infer_instrument_from_midi(midi_path: Path, codes: list[str]) -> str | None:
    """
    Try to infer the instrument code from filename or MIDI track names.
    Returns the code if it matches any in `codes`, else None.
    """
    stem = midi_path.stem
    # Check performer lookup
    for performer, code in _PERFORMER_INSTRUMENT.items():
        if performer.lower() in stem.lower():
            if code in codes:
                return code

    # Check MIDI track names for instrument keywords
    try:
        mid = pretty_midi.PrettyMIDI(str(midi_path))
        for instrument in mid.instruments:
            name = instrument.name.lower()
            if any(k in name for k in ("alto", "as", "alto sax")):
                if "as" in codes:
                    return "as"
            if any(k in name for k in ("tenor", "ts", "tenor sax")):
                if "ts" in codes:
                    return "ts"
            if any(k in name for k in ("soprano", "ss")):
                if "ss" in codes:
                    return "ss"
            if any(k in name for k in ("baritone", "bari", "bs")):
                if "bs" in codes:
                    return "bs"
            if any(k in name for k in ("trumpet", "tp")):
                if "tp" in codes:
                    return "tp"
            if any(k in name for k in ("trombone", "tb")):
                if "tb" in codes:
                    return "tb"
    except Exception:
        pass

    return None


def load_solos_from_midi(midi_dir: Path, instrument_codes: list[str]) -> list[dict]:
    """
    Load solos from unquantised MIDI files.
    Each MIDI file is expected to contain a single solo track (WJD convention).
    """
    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.MID"))
    if not midi_files:
        midi_files = sorted(midi_dir.rglob("*.mid"))

    solos = []
    skipped = 0

    for path in midi_files:
        code = _infer_instrument_from_midi(path, instrument_codes)
        if code is None:
            skipped += 1
            continue

        try:
            mid = pretty_midi.PrettyMIDI(str(path))
        except Exception as e:
            print(f"  Warning: could not parse {path.name}: {e}")
            continue

        tempo_bpm = mid.estimate_tempo() or 120.0

        # Collect all notes from all non-drum tracks
        notes = []
        for inst in mid.instruments:
            if inst.is_drum:
                continue
            for note in inst.notes:
                notes.append({
                    "pitch":  note.pitch,
                    "onset":  note.start,
                    "offset": note.end,
                })

        notes.sort(key=lambda n: n["onset"])

        solos.append({
            "melid":      path.stem,
            "title":      path.stem,
            "performer":  path.stem.split("_")[0] if "_" in path.stem else path.stem,
            "instrument": code,
            "tempo_bpm":  float(tempo_bpm),
            "notes":      notes,
        })

    if skipped:
        print(f"  {skipped} MIDI files skipped (instrument not matched to family).")

    return solos


# ---------------------------------------------------------------------------
# Shared extraction and saving
# ---------------------------------------------------------------------------

def _extract_and_save(family: str, solos: list[dict]) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not solos:
        print("No solos found for this instrument family.")
        sys.exit(1)

    print(f"Found {len(solos)} solos.")

    pitch_min, pitch_max = PITCH_RANGES[family]
    all_sequences = []
    all_chords    = []
    meta = []
    total_phrases = 0
    total_notes   = 0
    skipped_pitch = 0

    for solo in solos:
        phrases = segment_phrases(solo["notes"])
        for phrase in phrases:
            for n in phrase:
                if n["pitch"] < pitch_min or n["pitch"] > pitch_max:
                    skipped_pitch += 1

            tokens     = phrase_to_tokens(phrase, solo["tempo_bpm"])
            chord_seq  = phrase_to_chord_sequence(phrase)
            all_sequences.append(np.array(tokens,    dtype=np.int16))
            all_chords.append(   np.array(chord_seq, dtype=np.int8))
            meta.append({
                "melid":      solo["melid"],
                "title":      solo["title"],
                "performer":  solo["performer"],
                "instrument": solo["instrument"],
                "tempo_bpm":  solo["tempo_bpm"],
                "n_notes":    len(phrase),
            })
            total_phrases += 1
            total_notes   += len(phrase)

    out_seqs   = PROCESSED_DIR / f"{family}_sequences.npy"
    out_chords = PROCESSED_DIR / f"{family}_chords.npy"
    out_meta   = PROCESSED_DIR / f"{family}_meta.json"

    np.save(out_seqs,   np.array(all_sequences, dtype=object), allow_pickle=True)
    np.save(out_chords, np.array(all_chords,    dtype=object), allow_pickle=True)
    with open(out_meta, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'Solos processed:':30s} {len(solos)}")
    print(f"{'Phrases extracted:':30s} {total_phrases}")
    print(f"{'Total notes:':30s} {total_notes}")
    print(f"{'Notes outside pitch range:':30s} {skipped_pitch} (clipped, not dropped)")
    print(f"{'Vocab size:':30s} {VOCAB_SIZE}")
    print(f"\nSaved:")
    print(f"  {out_seqs}")
    print(f"  {out_chords}")
    print(f"  {out_meta}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(family: str, midi_only: bool = False) -> None:
    codes = codes_for(family)

    if midi_only:
        if not WJD_MIDI_DIR.exists():
            print(f"ERROR: MIDI directory not found at {WJD_MIDI_DIR}")
            print(f"Extract RELEASE2.0_mid_unquant.zip into {WJD_MIDI_DIR}")
            sys.exit(1)
        print(f"Loading {family} solos from MIDI files in {WJD_MIDI_DIR} ...")
        solos = load_solos_from_midi(WJD_MIDI_DIR, codes)
    else:
        if not WJD_DB.exists():
            print(f"ERROR: WJD database not found at {WJD_DB}")
            print("Download wjazzd.db from jazzomat.hfm-weimar.de and place it in data/raw/")
            print("Or use --midi-only to extract from MIDI files without the database.")
            sys.exit(1)
        print(f"Opening {WJD_DB}")
        db = sqlite3.connect(WJD_DB)
        print(f"Loading {family} solos (instrument codes: {codes}) ...")
        solos = load_solos(db, codes)
        db.close()

    _extract_and_save(family, solos)


def main():
    parser = argparse.ArgumentParser(description="Wolfson WJD data preparation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inspect", action="store_true",
                       help="Print database schema and instrument distribution (requires wjazzd.db)")
    group.add_argument("--instrument", metavar="FAMILY",
                       help="Instrument family to extract (sax, trumpet, trombone, flute)")
    parser.add_argument("--midi-only", action="store_true",
                        help="Extract from MIDI files only, without the SQLite database")
    args = parser.parse_args()

    if args.inspect:
        if not WJD_DB.exists():
            print(f"ERROR: Database not found at {WJD_DB}")
            sys.exit(1)
        db = sqlite3.connect(WJD_DB)
        inspect(db)
        db.close()
    else:
        prepare(args.instrument, midi_only=args.midi_only)


if __name__ == "__main__":
    main()

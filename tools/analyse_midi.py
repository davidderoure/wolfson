#!/usr/bin/env python3
"""
analyse_midi.py — stage-by-stage analysis of a Wolfson self-play MIDI recording.

Usage:
    python tools/analyse_midi.py path/to/recording.mid [--bpm 90] [--arc-secs 300]

Prints per-stage note duration statistics for the melody (sax) channel and a
three-row comparison table if multiple files are passed.

    python tools/analyse_midi.py 60bpm.mid 90bpm.mid 120bpm.mid
"""

import sys
import argparse
import statistics
from pathlib import Path

try:
    import mido
except ImportError:
    sys.exit("mido is required:  pip install mido")

# ---------------------------------------------------------------------------
# Stage boundaries as fractions of total arc duration
# ---------------------------------------------------------------------------
STAGES = [
    ("sparse",         0.00, 0.15),
    ("building",       0.15, 0.40),
    ("peak",           0.40, 0.65),
    ("recapitulation", 0.65, 0.85),
    ("resolution",     0.85, 1.00),
]

SHORT_THRESHOLD = 0.40   # beats — notes shorter than this count as "short"
LONG_THRESHOLD  = 0.75   # beats — notes longer than this count as "long"
REST_PITCH      = 0      # sentinel used by phrase_generator for rests


def analyse_file(path: Path, arc_secs: float = 300.0) -> dict:
    """Return per-stage and overall stats for the melody channel of a MIDI file."""
    mid = mido.MidiFile(str(path))

    # Resolve tempo
    tempo_us = 500_000  # default 120bpm
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo_us = msg.tempo
                break
        else:
            continue
        break

    bpm        = 60_000_000 / tempo_us
    ticks_beat = mid.ticks_per_beat
    arc_beats  = arc_secs * (bpm / 60.0)

    # Collect note events from the melody channel (channel 0)
    # Build a list of (start_beat, duration_beats, pitch) tuples.
    notes = []
    for track in mid.tracks:
        active = {}          # pitch -> start_tick
        tick   = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.channel == 0 and msg.velocity > 0:
                if msg.note != REST_PITCH:
                    active[msg.note] = tick
            elif msg.type in ("note_off", "note_on") and msg.channel == 0:
                if msg.note in active and msg.velocity == 0 or msg.type == "note_off":
                    if msg.note in active:
                        start_tick = active.pop(msg.note)
                        dur_beats  = (tick - start_tick) / ticks_beat
                        start_beat = start_tick / ticks_beat
                        if dur_beats > 0:
                            notes.append((start_beat, dur_beats, msg.note))

    if not notes:
        return {"path": path, "bpm": bpm, "stages": {}, "overall": None}

    # Assign each note to a stage
    stage_notes: dict[str, list[float]] = {s[0]: [] for s in STAGES}
    for start_beat, dur_beats, _ in notes:
        frac = start_beat / arc_beats
        for name, lo, hi in STAGES:
            if lo <= frac < hi:
                stage_notes[name].append(dur_beats)
                break
        else:
            # Beyond arc end — ignore
            pass

    def stats(durs: list[float]) -> dict:
        if not durs:
            return {"n": 0, "mean": 0, "median": 0, "short_pct": 0, "long_pct": 0}
        return {
            "n":         len(durs),
            "mean":      statistics.mean(durs),
            "median":    statistics.median(durs),
            "short_pct": 100 * sum(d < SHORT_THRESHOLD for d in durs) / len(durs),
            "long_pct":  100 * sum(d >= LONG_THRESHOLD  for d in durs) / len(durs),
        }

    stage_stats = {name: stats(stage_notes[name]) for name, *_ in STAGES}
    all_durs    = [d for _, d, _ in notes if d > 0]
    overall     = stats(all_durs)

    return {
        "path":    path,
        "bpm":     bpm,
        "n_notes": len(notes),
        "stages":  stage_stats,
        "overall": overall,
    }


def print_report(result: dict):
    p = result["path"]
    print(f"\n{'='*70}")
    print(f"  File : {p.name}")
    print(f"  BPM  : {result['bpm']:.1f}   Total melody notes: {result['n_notes']}")
    print(f"{'='*70}")

    header = f"  {'Stage':<18} {'N':>4}  {'Mean':>6}  {'Median':>6}  {'Short%':>7}  {'Long%':>6}"
    print(header)
    print("  " + "-" * 56)

    for name, *_ in STAGES:
        s = result["stages"][name]
        if s["n"] == 0:
            print(f"  {name:<18} {'—':>4}")
            continue
        print(
            f"  {name:<18} {s['n']:>4}  "
            f"{s['mean']:>6.3f}b  {s['median']:>6.3f}b  "
            f"{s['short_pct']:>6.1f}%  {s['long_pct']:>5.1f}%"
        )

    o = result["overall"]
    print("  " + "-" * 56)
    print(
        f"  {'OVERALL':<18} {o['n']:>4}  "
        f"{o['mean']:>6.3f}b  {o['median']:>6.3f}b  "
        f"{o['short_pct']:>6.1f}%  {o['long_pct']:>5.1f}%"
    )


def print_comparison(results: list[dict]):
    print(f"\n{'='*70}")
    print("  TEMPO COMPARISON  (overall melody stats)")
    print(f"{'='*70}")
    header = f"  {'File':<30} {'BPM':>5}  {'Mean':>6}  {'Short%':>7}  {'Long%':>6}"
    print(header)
    print("  " + "-" * 58)
    for r in results:
        o = r["overall"]
        print(
            f"  {r['path'].name:<30} {r['bpm']:>5.0f}  "
            f"{o['mean']:>6.3f}b  {o['short_pct']:>6.1f}%  {o['long_pct']:>5.1f}%"
        )


def main():
    parser = argparse.ArgumentParser(description="Analyse Wolfson self-play MIDI recordings.")
    parser.add_argument("files", nargs="+", type=Path, help="MIDI file(s) to analyse")
    parser.add_argument("--arc-secs", type=float, default=300.0,
                        help="Arc duration in seconds (default 300)")
    args = parser.parse_args()

    results = []
    for f in args.files:
        if not f.exists():
            print(f"WARNING: {f} not found, skipping", file=sys.stderr)
            continue
        r = analyse_file(f, arc_secs=args.arc_secs)
        print_report(r)
        results.append(r)

    if len(results) > 1:
        print_comparison(results)

    print()


if __name__ == "__main__":
    main()

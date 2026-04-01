#!/usr/bin/env python3
"""
analyse_midi.py — stage-by-stage analysis of a Wolfson self-play MIDI recording.

Usage:
    python tools/analyse_midi.py path/to/recording.mid [--arc-secs 300]

Prints per-stage note duration statistics for the melody (sax) channel and a
comparison table when multiple files are passed.

    python tools/analyse_midi.py 60bpm.mid 90bpm.mid 120bpm.mid

Add --plot to generate a PNG with rolling-window curves plotted against the
5-minute arc time axis.  One line per file; stage boundaries shaded.

    python tools/analyse_midi.py a.mid b.mid --plot
    python tools/analyse_midi.py a.mid b.mid --plot --plot-out my_comparison.png
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

# Subtle background colours for stage shading in plots
STAGE_SHADING = {
    "sparse":         "#cccccc",
    "building":       "#aaddee",
    "peak":           "#ffbbbb",
    "recapitulation": "#bbddbb",
    "resolution":     "#ffeebb",
}

SHORT_THRESHOLD = 0.40   # beats — notes shorter than this count as "short"
LONG_THRESHOLD  = 0.75   # beats — notes longer than this count as "long"
REST_PITCH      = 0      # sentinel used by phrase_generator for rests

# Rolling-window parameters for time-series plots
WINDOW_SECS = 40.0   # width of each rolling window
STEP_SECS   = 10.0   # spacing between window centres


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyse_file(path: Path, arc_secs: float = 300.0) -> dict:
    """Return per-stage stats and raw note list for the melody channel."""
    mid = mido.MidiFile(str(path))

    # Resolve tempo
    tempo_us = 500_000  # default 120 bpm
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
    secs_beat  = 60.0 / bpm

    # Collect note events from the melody channel (channel 0).
    # Each entry: (start_sec, start_beat, duration_beats, pitch)
    notes = []
    for track in mid.tracks:
        active = {}   # pitch -> start_tick
        tick   = 0
        for msg in track:
            tick += msg.time
            if msg.type == "note_on" and msg.channel == 0 and msg.velocity > 0:
                if msg.note != REST_PITCH:
                    active[msg.note] = tick
            elif msg.type in ("note_off", "note_on") and msg.channel == 0:
                if msg.note in active and (msg.velocity == 0 or msg.type == "note_off"):
                    if msg.note in active:
                        start_tick = active.pop(msg.note)
                        dur_beats  = (tick - start_tick) / ticks_beat
                        start_beat = start_tick / ticks_beat
                        start_sec  = start_beat * secs_beat
                        if dur_beats > 0:
                            notes.append((start_sec, start_beat, dur_beats, msg.note))

    if not notes:
        return {"path": path, "bpm": bpm, "n_notes": 0,
                "stages": {}, "overall": None, "notes_raw": []}

    # Assign each note to a stage
    stage_notes: dict[str, list[float]] = {s[0]: [] for s in STAGES}
    for _sec, start_beat, dur_beats, _ in notes:
        frac = start_beat / arc_beats
        for name, lo, hi in STAGES:
            if lo <= frac < hi:
                stage_notes[name].append(dur_beats)
                break

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
    all_durs    = [d for *_, d, _ in notes if d > 0]
    overall     = stats(all_durs)

    return {
        "path":      path,
        "bpm":       bpm,
        "n_notes":   len(notes),
        "stages":    stage_stats,
        "overall":   overall,
        # raw notes for time-series plotting: (start_sec, dur_beats, pitch)
        "notes_raw": [(s, d, p) for s, _b, d, p in notes],
    }


def compute_time_series(notes_raw: list, arc_secs: float,
                        window_secs: float = WINDOW_SECS,
                        step_secs:   float = STEP_SECS) -> dict:
    """
    Rolling-window stats across the performance arc.

    Returns a dict of equal-length lists:
        times       — window centre in minutes
        mean_dur    — mean note duration (beats) within window
        short_pct   — % notes shorter than SHORT_THRESHOLD
        note_rate   — notes per minute within window
    NaN used where a window contains no notes.
    """
    import math
    times, mean_durs, short_pcts, note_rates = [], [], [], []

    t = step_secs / 2.0
    while t <= arc_secs:
        t_lo = max(0.0, t - window_secs / 2.0)
        t_hi = min(arc_secs, t + window_secs / 2.0)
        w = [dur for start, dur, _ in notes_raw if t_lo <= start < t_hi]
        times.append(t / 60.0)   # minutes
        if w:
            mean_durs.append(statistics.mean(w))
            short_pcts.append(100.0 * sum(d < SHORT_THRESHOLD for d in w) / len(w))
            note_rates.append(len(w) / ((t_hi - t_lo) / 60.0))
        else:
            nan = math.nan
            mean_durs.append(nan)
            short_pcts.append(nan)
            note_rates.append(nan)
        t += step_secs

    return {
        "times":     times,
        "mean_dur":  mean_durs,
        "short_pct": short_pcts,
        "note_rate": note_rates,
    }


# ---------------------------------------------------------------------------
# Text output
# ---------------------------------------------------------------------------

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
    print("  COMPARISON  (overall melody stats)")
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


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

# Line colours for up to 8 files
_LINE_COLOURS = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#e377c2", "#17becf",
]


def plot_comparison(results: list[dict], arc_secs: float, out_path: Path):
    """
    Three-panel time-series plot: mean duration, short%, note rate.
    One line per file; stage regions shaded in background.
    Saved to out_path as PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit("matplotlib is required for --plot:  pip install matplotlib")

    arc_mins = arc_secs / 60.0

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("Wolfson self-play — arc time series", fontsize=14, fontweight="bold")

    panels = [
        (axes[0], "mean_dur",  "Mean note duration (beats)", None),
        (axes[1], "short_pct", f"Short notes  (< {SHORT_THRESHOLD}b)  %", (0, 100)),
        (axes[2], "note_rate", "Note rate  (notes / min)",  None),
    ]

    # Draw stage shading on every panel
    for ax, _key, _label, _ylim in panels:
        for name, lo, hi in STAGES:
            ax.axvspan(lo * arc_mins, hi * arc_mins,
                       color=STAGE_SHADING[name], alpha=0.35, zorder=0)
        # Stage name annotations on top panel only
    for name, lo, hi in STAGES:
        mid_x = (lo + hi) / 2 * arc_mins
        axes[0].text(mid_x, 1.01, name, transform=axes[0].get_xaxis_transform(),
                     ha="center", va="bottom", fontsize=8, color="#555555")

    for idx, result in enumerate(results):
        colour = _LINE_COLOURS[idx % len(_LINE_COLOURS)]
        label  = result["path"].stem
        ts     = compute_time_series(result["notes_raw"], arc_secs)

        for ax, key, _label, _ylim in panels:
            ax.plot(ts["times"], ts[key],
                    color=colour, linewidth=1.8, label=label, zorder=2)
            # Mark stage-boundary transitions with faint vertical lines
            for _name, lo, _hi in STAGES[1:]:   # skip first (starts at 0)
                ax.axvline(lo * arc_mins, color="#999999", linewidth=0.6,
                           linestyle="--", zorder=1)

    for ax, _key, ylabel, ylim in panels:
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", linewidth=0.4, alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ylim:
            ax.set_ylim(*ylim)

    axes[2].set_xlabel("Time (minutes)", fontsize=10)
    axes[2].set_xlim(0, arc_mins)

    # Single legend outside the top panel
    handles, labels = axes[0].get_legend_handles_labels()
    # De-duplicate (one entry per file, not per panel line)
    seen = {}
    for h, l in zip(handles, labels):
        seen[l] = h
    axes[0].legend(seen.values(), seen.keys(),
                   loc="upper right", fontsize=8, framealpha=0.8)

    plt.tight_layout()
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyse Wolfson self-play MIDI recordings.")
    parser.add_argument("files", nargs="+", type=Path,
                        help="MIDI file(s) to analyse")
    parser.add_argument("--arc-secs", type=float, default=300.0,
                        help="Arc duration in seconds (default 300)")
    parser.add_argument("--plot", action="store_true",
                        help="Generate a time-series plot (requires matplotlib)")
    parser.add_argument("--plot-out", type=Path, default=None,
                        help="Output path for plot PNG (default: wolfson_analysis.png)")
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

    if args.plot and results:
        out = args.plot_out or Path("wolfson_analysis.png")
        plot_comparison(results, arc_secs=args.arc_secs, out_path=out)

    print()


if __name__ == "__main__":
    main()

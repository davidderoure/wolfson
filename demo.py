"""
demo.py — Feature-focused testing without the 5-minute arc.

Each demo mode wires bass → PhraseGenerator with fixed, explicit parameters
so you can hear one feature at a time. The arc, leadership logic, and phrase
memory are all bypassed.

Usage
-----
    python demo.py --demo scale         # D Dorian scale bias
    python demo.py --demo swing         # triplet response to your playing
    python demo.py --demo straight      # no swing bias (compare with above)
    python demo.py --demo contour       # alternating ascending / descending endings
    python demo.py --demo progression   # ii-V-I in C, advancing each phrase
    python demo.py --demo blues         # 12-bar blues progression
    python demo.py --demo tritone       # always apply tritone substitution on V7
    python demo.py --demo pedal         # D pedal tone with cycling upper harmony
    python demo.py --demo free          # chromatic / unconstrained (baseline)

Global options (can combine with any demo):
    --root NOTE     tonic / modal root as a note name, e.g. --root D  (default: D)
    --temp FLOAT    generation temperature (default: 0.9)
    --notes INT     max notes per response phrase (default: 12)
    --port-in INT   MIDI input port index (default: from config.py)
    --port-out INT  MIDI output port index (default: from config.py)

Each phrase you play on bass triggers one sax response. The console prints
exactly which parameters were used so you know what you're hearing.
Press Ctrl-C to stop.
"""

import argparse
import threading
import time

from input.midi_listener   import MidiListener
from input.phrase_detector import PhraseDetector
from input.beat_estimator  import BeatEstimator
from input.phrase_analyzer import analyze
from generator.phrase_generator import PhraseGenerator
from output.midi_output    import MidiOutput
from controller.harmony    import HarmonyController, NAMED_PROGRESSIONS
from data.chords           import NC_INDEX, N_QUALITIES, QUAL_DOM
from data.scales           import scale_pitch_classes, chord_root, chord_to_mode
from config import (
    DEFAULT_INSTRUMENT, TEMPO_HINT_BPM,
    MIDI_INPUT_PORT, MIDI_OUTPUT_PORT,
)


# ---------------------------------------------------------------------------
# Note-name → pitch-class
# ---------------------------------------------------------------------------

NOTE_NAMES = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "F": 5, "F#": 6, "Gb": 6, "G": 7, "G#": 8,
    "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11,
}


def parse_root(name: str) -> int:
    pc = NOTE_NAMES.get(name.strip())
    if pc is None:
        raise ValueError(f"Unknown note name: {name!r}. Use C, D, Eb, F#, etc.")
    return pc


# ---------------------------------------------------------------------------
# Demo mode definitions
# ---------------------------------------------------------------------------

def build_demo(args) -> dict:
    """
    Return a fixed-parameter dict that will be used for every sax response.
    Dynamic fields (swing_bias when auto, contour when cycling) are handled
    at response time via callables stored here.
    """
    root = parse_root(args.root)
    demo = args.demo

    # Shared defaults
    cfg = {
        "temperature":         args.temp,
        "n_notes":             args.notes,
        "chord_idx":           NC_INDEX,
        "scale_pitch_classes": None,          # frozenset or None
        "swing_bias":          0.0,
        "contour_target":      "neutral",
        "description":         "",
        # harmony controller: only used in progression / blues / tritone / pedal
        "harmony":             None,
        # contour_cycle: if True, alternate ascending/descending per phrase
        "contour_cycle":       False,
        # swing_auto: if True, compute swing_bias from bass phrase features
        "swing_auto":          False,
    }

    if demo == "free":
        cfg["description"] = "free / chromatic — no scale, no swing bias, neutral contour"

    elif demo == "scale":
        pc = scale_pitch_classes(root, "dorian")
        from data.chords import QUAL_MINOR
        cid = root * N_QUALITIES + QUAL_MINOR
        cfg["chord_idx"]           = cid
        cfg["scale_pitch_classes"] = pc
        cfg["description"] = (
            f"{args.root} Dorian — scale pitch bias active, "
            f"scale pcs = {sorted(pc)}"
        )

    elif demo == "swing":
        cfg["swing_bias"]  = 1.0
        cfg["description"] = "swing — maximum triplet-grid duration bias (swing_bias=1.0)"

    elif demo == "straight":
        cfg["swing_bias"]  = 0.0
        cfg["description"] = "straight — no swing bias (swing_bias=0.0)"

    elif demo == "contour":
        cfg["contour_cycle"] = True
        cfg["description"]   = "contour — alternates ascending / descending phrase endings"

    elif demo == "progression":
        h = HarmonyController()
        h.set_mode("progression", key_root=root, prog_name="ii_v_i")
        cfg["harmony"]     = h
        cfg["description"] = f"ii-V-I in {args.root} — chord advances each phrase, tritone subs ~35%"

    elif demo == "blues":
        h = HarmonyController()
        h.set_mode("progression", key_root=root, prog_name="blues")
        cfg["harmony"]     = h
        cfg["description"] = f"12-bar blues in {args.root}"

    elif demo == "tritone":
        # Always tritone-sub the V7 in a ii-V-I by forcing TRITONE_SUB_PROB=1.0
        # We do this by patching harmony after construction.
        import controller.harmony as _hmod
        orig = _hmod.TRITONE_SUB_PROB
        _hmod.TRITONE_SUB_PROB = 1.0
        h = HarmonyController()
        h.set_mode("progression", key_root=root, prog_name="ii_v_i")
        _hmod.TRITONE_SUB_PROB = orig
        cfg["harmony"]     = h
        cfg["description"] = (
            f"tritone substitution — ii-V-I in {args.root} with V7 always replaced "
            f"by its tritone sub (bII7)"
        )

    elif demo == "pedal":
        h = HarmonyController()
        h.set_mode("pedal", pedal_root=root)
        cfg["harmony"]     = h
        cfg["description"] = f"{args.root} pedal tone — upper harmony cycles i, bVII7, i, V7"

    elif demo == "modal":
        pc = scale_pitch_classes(root, "dorian")
        from data.chords import QUAL_MINOR
        cid = root * N_QUALITIES + QUAL_MINOR
        cfg["chord_idx"]           = cid
        cfg["scale_pitch_classes"] = pc
        cfg["description"] = f"{args.root} Dorian modal — same as 'scale' demo"

    else:
        raise ValueError(f"Unknown demo mode: {demo!r}")

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Wolfson feature demo — no arc")
    parser.add_argument("--demo",     default="free",
                        choices=["free","scale","swing","straight","contour",
                                 "progression","blues","tritone","pedal","modal"],
                        help="Which feature to demo")
    parser.add_argument("--root",     default="D",  help="Tonic / modal root note name")
    parser.add_argument("--temp",     type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--notes",    type=int,   default=12,  help="Max notes per phrase")
    parser.add_argument("--port-in",  type=int,   default=MIDI_INPUT_PORT,  dest="port_in")
    parser.add_argument("--port-out", type=int,   default=MIDI_OUTPUT_PORT, dest="port_out")
    args = parser.parse_args()

    cfg       = build_demo(args)
    generator = PhraseGenerator(instrument=DEFAULT_INSTRUMENT)
    midi_out  = MidiOutput()
    beats     = BeatEstimator(initial_bpm=120.0, hint_bpm=TEMPO_HINT_BPM)
    _sax_lock = threading.Lock()

    _contour_state = {"flip": False}   # shared mutable state for contour cycling

    print(f"\nWolfson demo — {args.demo}")
    print(f"  {cfg['description']}")
    print(f"  temperature={args.temp}  max_notes={args.notes}  root={args.root}")
    print("\nPlay bass phrases. Ctrl-C to stop.\n")

    def on_bass_phrase(phrase: list[dict]):
        if not _sax_lock.acquire(blocking=False):
            return   # sax already playing

        try:
            features = analyze(phrase)

            # Resolve per-phrase dynamic parameters
            chord_idx  = cfg["chord_idx"]
            scale_pcs  = cfg["scale_pitch_classes"]
            swing_bias = cfg["swing_bias"]
            contour    = cfg["contour_target"]

            if cfg["harmony"] is not None:
                chord_idx, scale_pcs = cfg["harmony"].next_chord()

            if cfg["contour_cycle"]:
                contour = "ascending" if _contour_state["flip"] else "descending"
                _contour_state["flip"] = not _contour_state["flip"]

            notes = generator.generate(
                seed_phrase         = phrase,
                n_notes             = cfg["n_notes"],
                temperature         = cfg["temperature"],
                contour_target      = contour,
                chord_idx           = chord_idx,
                swing_bias          = swing_bias,
                scale_pitch_classes = scale_pcs,
            )
            if not notes:
                return

            beat_dur_sec  = beats.beat_duration
            durations_sec = [n["duration_beats"] * beat_dur_sec for n in notes]

            _print_response(args.demo, chord_idx, scale_pcs, swing_bias, contour,
                            notes, beats.bpm, features)

            midi_out.play_phrase(
                pitches   = [n["pitch"] for n in notes],
                durations = durations_sec,
            )
        finally:
            _sax_lock.release()

    def _note_on_hook(pitch, velocity, t):
        beats.note_on(t)
        detector.note_on(pitch, velocity, t)

    detector = PhraseDetector(on_phrase_complete=on_bass_phrase)
    listener = MidiListener(
        on_note_on  = _note_on_hook,
        on_note_off = detector.note_off,
    )

    midi_out.start()
    listener.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        listener.stop()
        midi_out.silence()
        midi_out.stop()


# ---------------------------------------------------------------------------
# Console output
# ---------------------------------------------------------------------------

def _print_response(demo, chord_idx, scale_pcs, swing_bias, contour, notes, bpm, features):
    from data.chords import N_QUALITIES
    NOTE_NAMES_REV = ["C","C#","D","Eb","E","F","F#","G","Ab","A","Bb","B"]

    chord_str = "NC"
    if chord_idx != NC_INDEX:
        r = chord_idx // N_QUALITIES
        q = chord_idx % N_QUALITIES
        q_str = {0:"maj", 1:"7", 2:"m7", 3:"dim"}.get(q, "?")
        chord_str = NOTE_NAMES_REV[r % 12] + q_str

    scale_str = f"{len(scale_pcs)} tones" if scale_pcs else "chromatic"
    feel      = features.get("rhythmic_feel", "?")
    swing_r   = features.get("swing_ratio", 0.0)

    print(
        f"[{demo:<12s}]  {bpm:5.1f} bpm  "
        f"chord={chord_str:<6s}  scale={scale_str:<12s}  "
        f"swing_bias={swing_bias:.1f}  contour={contour:<10s}  "
        f"bass={feel}({swing_r:.1f})  n={len(notes)}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
echo_bass.py — Echo bass phrases back on the sax output channel.

Used for setup verification and performance warm-up. Each detected bass
phrase is replayed note-for-note on the sax output with the original
pitches, velocities, and relative timing. No model loading required.

Usage:
    python tools/echo_bass.py
    python tools/echo_bass.py --channel 2     # different output channel
    python tools/echo_bass.py --transpose 12  # octave up
    python tools/echo_bass.py --delay 0.5     # pause before replay (seconds)
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import rtmidi

from config import MIDI_INPUT_PORT, MIDI_OUTPUT_PORT, MIDI_PITCH_MIN, MIDI_PITCH_MAX, MIDI_VELOCITY_MIN
from input.phrase_detector import PhraseDetector


# ---------------------------------------------------------------------------
# Playback
# ---------------------------------------------------------------------------

def play_phrase(midi_out, phrase, channel, transpose, delay):
    """Replay a phrase preserving relative timing and velocity."""
    if not phrase:
        return

    if delay > 0:
        time.sleep(delay)

    ch   = channel - 1          # rtmidi uses 0-indexed channels
    t0   = phrase[0]["onset"]
    wall = time.time()

    for note in phrase:
        # Wait until this note's scheduled time
        target = wall + (note["onset"] - t0)
        gap    = target - time.time()
        if gap > 0:
            time.sleep(gap)

        pitch = max(0, min(127, note["pitch"] + transpose))
        vel   = max(1, min(127, note.get("velocity", 64)))

        midi_out.send_message([0x90 | ch, pitch, vel])

        # Schedule note-off at 85% of the note's duration (standard articulation)
        dur = (note["offset"] - note["onset"]) * 0.85
        def _off(p=pitch, d=dur):
            time.sleep(max(0.02, d))
            midi_out.send_message([0x80 | ch, p, 0])
        threading.Thread(target=_off, daemon=True).start()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Echo bass phrases on the sax output channel."
    )
    parser.add_argument(
        "--channel", type=int, default=1, metavar="CH",
        help="Output MIDI channel (default: 1)",
    )
    parser.add_argument(
        "--transpose", type=int, default=0, metavar="N",
        help="Transpose in semitones, e.g. 12 for an octave up (default: 0)",
    )
    parser.add_argument(
        "--delay", type=float, default=0.0, metavar="SEC",
        help="Pause before replaying each phrase, seconds (default: 0)",
    )
    args = parser.parse_args()

    # --- Output ---
    midi_out  = rtmidi.MidiOut()
    out_ports = midi_out.get_ports()
    print("Output ports:")
    for i, name in enumerate(out_ports):
        marker = " ◄" if i == MIDI_OUTPUT_PORT else ""
        print(f"  [{i}] {name}{marker}")
    if MIDI_OUTPUT_PORT >= len(out_ports):
        print(f"\nERROR: MIDI_OUTPUT_PORT={MIDI_OUTPUT_PORT} not available.")
        sys.exit(1)
    midi_out.open_port(MIDI_OUTPUT_PORT)

    # --- Input ---
    midi_in  = rtmidi.MidiIn()
    in_ports = midi_in.get_ports()
    print("\nInput ports:")
    for i, name in enumerate(in_ports):
        marker = " ◄" if i == MIDI_INPUT_PORT else ""
        print(f"  [{i}] {name}{marker}")
    if MIDI_INPUT_PORT >= len(in_ports):
        print(f"\nERROR: MIDI_INPUT_PORT={MIDI_INPUT_PORT} not available.")
        sys.exit(1)

    # --- Phrase detector ---
    def on_phrase(phrase):
        pitches = [n["pitch"] for n in phrase]
        vels    = [n.get("velocity", 64) for n in phrase]
        print(
            f"  phrase: {len(phrase)} notes  "
            f"pitches {pitches}  "
            f"vel {min(vels)}–{max(vels)}"
        )
        threading.Thread(
            target=play_phrase,
            args=(midi_out, phrase, args.channel, args.transpose, args.delay),
            daemon=True,
        ).start()

    detector = PhraseDetector(on_phrase_complete=on_phrase)

    def on_midi(message, _data=None):
        msg, _dt = message
        if len(msg) < 2:
            return
        status = msg[0] & 0xF0
        pitch  = msg[1]
        vel    = msg[2] if len(msg) > 2 else 0
        if not (MIDI_PITCH_MIN <= pitch <= MIDI_PITCH_MAX):
            return
        t = time.time()
        if status == 0x90 and vel > 0:
            if vel < MIDI_VELOCITY_MIN:
                return
            detector.note_on(pitch, vel, t)
        elif status == 0x80 or (status == 0x90 and vel == 0):
            detector.note_off(pitch, t)

    midi_in.open_port(MIDI_INPUT_PORT)
    midi_in.set_callback(on_midi)
    midi_in.ignore_types(sysex=True, timing=True, active_sense=True)

    print(
        f"\nReady — echoing bass → ch {args.channel}  "
        f"transpose={args.transpose:+d}  delay={args.delay}s"
    )
    print("Play bass phrases. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        midi_in.close_port()
        midi_out.close_port()
        print("\nStopped.")


if __name__ == "__main__":
    main()

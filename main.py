"""
Wolfson — interactive jazz bass + sax improvisation system.

Pipeline:
  MidiListener → PhraseDetector  → PhraseAnalyzer
                      │
                 BeatEstimator   (tap tempo from bass onsets)
                      │
                 ArcController   (5-min arc, leadership, proactive mode)
                      │
                 PhraseGenerator (LSTM + contour steering + chord conditioning)
                      │
                 MidiOutput      (per-note duration, articulation)

Proactive thread: fires every PROACTIVE_CHECK_INTERVAL seconds; if the arc
controller decides the sax should initiate, it generates and plays a phrase
without waiting for a bass phrase to complete.

Self-play mode (--self-play):
  The system feeds its own sax output back as input, creating an autonomous
  generative loop. No MIDI hardware needed. A short seed phrase bootstraps
  the first exchange; thereafter the sax continuously responds to itself.
  The 5-minute structural arc still governs the performance.
"""

import argparse
import threading
import time

from input.midi_listener    import MidiListener
from input.phrase_detector  import PhraseDetector
from input.phrase_analyzer  import analyze
from input.beat_estimator   import BeatEstimator
from memory.phrase_memory   import PhraseMemory
from generator.phrase_generator import PhraseGenerator
from controller.arc_controller  import ArcController
from output.midi_output     import MidiOutput
from config import DEFAULT_INSTRUMENT, TEMPO_HINT_BPM

PROACTIVE_CHECK_INTERVAL = 0.5   # seconds between proactive checks

# Self-play: brief silence between phrases (seconds) — musical breathing room
SELF_PLAY_PHRASE_GAP = 0.05

# Self-play seed: D minor pentatonic opening motif (pitch, beat-duration pairs)
# Gives the LSTM something musical to respond to on the very first exchange.
_SEED_PITCHES    = [62, 65, 67, 69, 72, 69, 67]   # D4 F4 G4 A4 C5 A4 G4
_SEED_DUR_BEATS  = [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0]


def main():
    parser = argparse.ArgumentParser(
        description="Wolfson interactive jazz improvisation system"
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="Autonomous mode: sax feeds its output back as its own input. "
             "No MIDI hardware required.",
    )
    parser.add_argument(
        "--bpm", type=float, default=120.0,
        help="Tempo for self-play mode (default: 120)",
    )
    args = parser.parse_args()

    self_play   = args.self_play
    initial_bpm = args.bpm

    memory    = PhraseMemory()
    generator = PhraseGenerator(instrument=DEFAULT_INSTRUMENT)
    arc       = ArcController(memory)
    midi_out  = MidiOutput()
    beats     = BeatEstimator(
        initial_bpm = initial_bpm,
        hint_bpm    = TEMPO_HINT_BPM or initial_bpm,
    )

    _running  = threading.Event()
    _running.set()
    _sax_lock = threading.Lock()   # one sax phrase at a time

    # ------------------------------------------------------------------
    # Bass phrase handler (reactive path)
    # ------------------------------------------------------------------

    def on_bass_phrase(phrase: list[dict]):
        memory.store(phrase, source="bass")
        params = arc.on_bass_phrase(phrase)
        _respond(params, triggered_by="bass")

    # ------------------------------------------------------------------
    # Shared response path
    # ------------------------------------------------------------------

    def _respond(params: dict, triggered_by: str):
        if params is None:
            return
        if not _sax_lock.acquire(blocking=False):
            return   # sax already mid-phrase

        notes_out    = []
        beat_dur_sec = beats.beat_duration

        try:
            notes = generator.generate(
                seed_phrase         = params["seed"],
                n_notes             = params["n_notes"],
                temperature         = params["temperature"],
                contour_target      = params["contour_target"],
                chord_idx           = params["chord_idx"],
                swing_bias          = params.get("swing_bias", 0.0),
                scale_pitch_classes = params.get("scale_pitch_classes"),
            )
            if not notes:
                # Model generated nothing (END_TOKEN sampled immediately).
                # In self-play, re-inject the seed so the loop doesn't stall.
                if self_play and _running.is_set():
                    beat_dur_sec = beats.beat_duration
                    _schedule_feedback(
                        [{"pitch":          n["pitch"],
                          "duration_beats": (n["offset"] - n["onset"])
                                            / beat_dur_sec}
                         for n in params["seed"]],
                        beat_dur_sec,
                    )
                return

            beat_dur_sec  = beats.beat_duration
            durations_sec = [n["duration_beats"] * beat_dur_sec for n in notes]

            memory.store(
                [{"pitch": n["pitch"], "velocity": 80,
                  "onset": 0, "offset": n["duration_beats"]}
                 for n in notes],
                source="sax",
            )

            _log(params, triggered_by, notes, beats.bpm)
            midi_out.play_phrase(
                pitches   = [n["pitch"] for n in notes],
                durations = durations_sec,
                velocity  = params.get("velocity", 80),
            )

            notes_out = notes   # capture after successful play

        finally:
            arc.on_sax_played()
            _sax_lock.release()

        # Self-play feedback — lock already released before this runs
        if self_play and notes_out and _running.is_set():
            _schedule_feedback(notes_out, beat_dur_sec)

    # ------------------------------------------------------------------
    # Self-play feedback
    # ------------------------------------------------------------------

    def _schedule_feedback(notes: list[dict], beat_dur_sec: float):
        """
        Inject sax notes back as a bass phrase in a new daemon thread.
        A short gap simulates the silence between phrases.
        """
        def _run():
            time.sleep(SELF_PLAY_PHRASE_GAP)
            if not _running.is_set():
                return
            now    = time.time()
            phrase = []
            onset  = 0.0
            for n in notes:
                dur_sec = n["duration_beats"] * beat_dur_sec
                phrase.append({
                    "pitch":    n["pitch"],
                    "velocity": 64,
                    "onset":    now + onset,
                    "offset":   now + onset + dur_sec,
                })
                beats.note_on(now + onset)   # keep tempo estimator warm
                onset += dur_sec
            on_bass_phrase(phrase)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Proactive background thread
    # ------------------------------------------------------------------

    def _proactive_loop():
        while _running.is_set():
            time.sleep(PROACTIVE_CHECK_INTERVAL)
            if arc.should_play_proactively():
                params = arc.get_proactive_params()
                if params:
                    _respond(params, triggered_by="sax")

    # ------------------------------------------------------------------
    # MIDI I/O and note-on hook for beat estimation
    # ------------------------------------------------------------------

    def _note_on_with_beat(pitch, velocity, t):
        beats.note_on(t)
        detector.note_on(pitch, velocity, t)

    detector = PhraseDetector(on_phrase_complete=on_bass_phrase)
    listener = MidiListener(
        on_note_on  = _note_on_with_beat,
        on_note_off = detector.note_off,
    )

    midi_out.start()
    arc.start()

    if self_play:
        # Seed the loop with an opening phrase in a background thread
        # so main() is not blocked before the KeyboardInterrupt handler.
        def _bootstrap():
            time.sleep(0.2)   # allow arc to initialise
            if not _running.is_set():
                return
            beat_dur = 60.0 / initial_bpm
            now      = time.time()
            phrase   = []
            onset    = 0.0
            for p, d in zip(_SEED_PITCHES, _SEED_DUR_BEATS):
                dur_sec = d * beat_dur
                phrase.append({
                    "pitch":    p,
                    "velocity": 64,
                    "onset":    now + onset,
                    "offset":   now + onset + dur_sec,
                })
                beats.note_on(now + onset)
                onset += dur_sec
            on_bass_phrase(phrase)

        threading.Thread(target=_bootstrap, daemon=True).start()
        print(
            f"Wolfson self-play mode. {initial_bpm:.0f} BPM. "
            "Ctrl-C to stop.\n"
        )
    else:
        listener.start()
        print("Wolfson ready. Play bass. Ctrl-C to stop.\n")

    proactive_thread = threading.Thread(target=_proactive_loop, daemon=True)
    proactive_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        _running.clear()
        if not self_play:
            listener.stop()
        midi_out.silence()
        midi_out.stop()


# ------------------------------------------------------------------
# Console logging
# ------------------------------------------------------------------

def _log(params: dict, triggered_by: str, notes: list, bpm: float):
    stage   = params.get("stage", "?")
    lead    = params.get("leadership", "?")
    mode    = params.get("mode", "?")
    contour = params.get("contour_target", "?")
    harm    = params.get("harmonic_mode", "?")
    src     = params.get("scale_source", "arc")
    vel     = params.get("velocity", 80)
    print(
        f"[{stage:>14s}]  {bpm:5.1f} bpm  lead={lead:<3s}  "
        f"trigger={triggered_by:<4s}  mode={mode:<8s}  "
        f"harm={harm:<12s}  scale={src:<5s}  contour={contour:<10s}  vel={vel:3d}  n={len(notes)}"
    )


if __name__ == "__main__":
    main()

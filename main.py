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
"""

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


def main():
    memory    = PhraseMemory()
    generator = PhraseGenerator(instrument=DEFAULT_INSTRUMENT)
    arc       = ArcController(memory)
    midi_out  = MidiOutput()
    beats     = BeatEstimator(initial_bpm=120.0, hint_bpm=TEMPO_HINT_BPM)

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

        try:
            notes = generator.generate(
                seed_phrase    = params["seed"],
                n_notes        = params["n_notes"],
                temperature    = params["temperature"],
                contour_target = params["contour_target"],
                chord_idx      = params["chord_idx"],
                swing_bias     = params.get("swing_bias", 0.0),
            )
            if not notes:
                return

            # Convert beat durations → seconds using live tempo estimate
            beat_dur_sec = beats.beat_duration
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
            )

        finally:
            arc.on_sax_played()
            _sax_lock.release()

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
    listener.start()

    proactive_thread = threading.Thread(target=_proactive_loop, daemon=True)
    proactive_thread.start()

    print("Wolfson ready. Play bass. Ctrl-C to stop.\n")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        _running.clear()
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
    print(
        f"[{stage:>14s}]  {bpm:5.1f} bpm  lead={lead:<3s}  "
        f"trigger={triggered_by:<4s}  mode={mode:<8s}  "
        f"contour={contour:<10s}  n={len(notes)}"
    )


if __name__ == "__main__":
    main()

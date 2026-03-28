"""
Wolfson — interactive jazz bass + sax improvisation system.

Pipeline:
  MidiListener → PhraseDetector → PhraseAnalyzer
                                       │
                                  ArcController ← tracks time, leadership, proactive state
                                       │
                                  PhraseGenerator ← LSTM + contour steering + chord conditioning
                                       │
                                  MidiOutput → synth

Proactive mode: a background thread checks periodically whether the sax should
initiate a phrase (rather than waiting for a bass phrase to complete).
"""

import threading
import time

from input.midi_listener   import MidiListener
from input.phrase_detector import PhraseDetector
from input.phrase_analyzer import analyze
from memory.phrase_memory  import PhraseMemory
from generator.phrase_generator import PhraseGenerator
from controller.arc_controller  import ArcController
from output.midi_output    import MidiOutput
from config import DEFAULT_INSTRUMENT

PROACTIVE_CHECK_INTERVAL = 0.5   # seconds between proactive checks


def main():
    memory    = PhraseMemory()
    generator = PhraseGenerator(instrument=DEFAULT_INSTRUMENT)
    arc       = ArcController(memory)
    midi_out  = MidiOutput()

    _running = threading.Event()
    _running.set()
    _sax_lock = threading.Lock()   # prevent overlapping sax phrases

    # ------------------------------------------------------------------
    # Bass phrase handler (reactive path)
    # ------------------------------------------------------------------

    def on_bass_phrase(phrase: list[dict]):
        memory.store(phrase, source="bass")
        params = arc.on_bass_phrase(phrase)
        _respond(params, triggered_by="bass")

    # ------------------------------------------------------------------
    # Sax response (shared by reactive and proactive paths)
    # ------------------------------------------------------------------

    def _respond(params: dict, triggered_by: str):
        if params is None:
            return

        if not _sax_lock.acquire(blocking=False):
            return   # sax already playing, skip

        try:
            notes = generator.generate(
                seed_phrase    = params["seed"],
                n_notes        = params["n_notes"],
                temperature    = params["temperature"],
                contour_target = params["contour_target"],
                chord_idx      = params["chord_idx"],
            )

            if not notes:
                return

            memory.store(
                [{"pitch": n["pitch"], "velocity": 80,
                  "onset": 0, "offset": n["duration_beats"]}
                 for n in notes],
                source="sax",
            )

            _log(params, triggered_by, len(notes))

            # Play in this thread (lock held throughout so proactive path
            # can't interrupt); release when done
            _play(notes, params)

        finally:
            arc.on_sax_played()
            _sax_lock.release()

    def _play(notes: list[dict], params: dict):
        """Convert {pitch, duration_beats} to MIDI, using a simple tempo estimate."""
        tempo_bpm = 120.0   # TODO: derive from live bass phrase timing
        beat_dur  = 60.0 / tempo_bpm
        midi_out.play_phrase(
            pitches  = [n["pitch"] for n in notes],
            duration = beat_dur * sum(n["duration_beats"] for n in notes) / max(len(notes), 1),
        )

    # ------------------------------------------------------------------
    # Proactive background thread
    # ------------------------------------------------------------------

    def _proactive_loop():
        while _running.is_set():
            time.sleep(PROACTIVE_CHECK_INTERVAL)
            if arc.should_play_proactively():
                params = arc.get_proactive_params()
                if params:
                    _respond(params, triggered_by="sax (proactive)")

    # ------------------------------------------------------------------
    # MIDI I/O setup
    # ------------------------------------------------------------------

    detector = PhraseDetector(on_phrase_complete=on_bass_phrase)
    listener = MidiListener(
        on_note_on  = detector.note_on,
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
        midi_out.stop()


# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------

def _log(params: dict, triggered_by: str, n_notes: int):
    elapsed  = params.get("_elapsed", 0)
    stage    = params.get("stage", "?")
    lead     = params.get("leadership", "?")
    mode     = params.get("mode", "?")
    contour  = params.get("contour_target", "?")
    print(
        f"[{stage:>14s}]  lead={lead:<3s}  trigger={triggered_by:<20s}"
        f"  mode={mode:<8s}  contour={contour:<10s}  n={n_notes}"
    )


if __name__ == "__main__":
    main()

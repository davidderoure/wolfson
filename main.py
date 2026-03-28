"""
Wolfson — interactive jazz bass + sax improvisation system.

Architecture:
  MidiListener -> PhraseDetector -> ArcController -> PhraseGenerator -> MidiOutput
                                         |
                                    PhraseMemory
"""

import threading
import time

from input.midi_listener import MidiListener
from input.phrase_detector import PhraseDetector
from memory.phrase_memory import PhraseMemory
from generator.phrase_generator import PhraseGenerator
from controller.arc_controller import ArcController
from output.midi_output import MidiOutput


def main():
    memory = PhraseMemory()
    generator = PhraseGenerator()   # pass model_path= once trained
    arc = ArcController(memory)
    midi_out = MidiOutput()

    def on_bass_phrase(phrase):
        memory.store(phrase, source="bass")
        params = arc.get_response_params(phrase)
        print(f"[{arc.stage():>14s}] {arc.elapsed():5.1f}s  mode={params['mode']}  n={params['n_notes']}")

        pitches = generator.generate(
            seed_phrase=params["seed"],
            n_notes=params["n_notes"],
            temperature=params["temperature"],
        )

        if pitches:
            memory.store(
                [{"pitch": p, "velocity": 80, "onset": 0, "offset": 0} for p in pitches],
                source="sax",
            )
            # Play in a thread so we don't block phrase detection
            t = threading.Thread(target=midi_out.play_phrase, args=(pitches,), daemon=True)
            t.start()

    detector = PhraseDetector(on_phrase_complete=on_bass_phrase)
    listener = MidiListener(
        on_note_on=detector.note_on,
        on_note_off=detector.note_off,
    )

    midi_out.start()
    arc.start()
    listener.start()

    print("Wolfson ready. Play bass. Ctrl-C to stop.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        listener.stop()
        midi_out.stop()


if __name__ == "__main__":
    main()

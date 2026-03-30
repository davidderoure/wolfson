"""Sends generated sax phrases to a MIDI output port."""

import rtmidi
import time
from config import MIDI_OUTPUT_PORT

DEFAULT_VELOCITY    = 80
ARTICULATION_RATIO  = 0.85   # note sounds for this fraction of its slot; rest is silence


class MidiOutput:
    def __init__(self):
        self._midi_out = rtmidi.MidiOut()

    def start(self):
        ports = self._midi_out.get_ports()
        if not ports:
            raise RuntimeError("No MIDI output ports found.")
        print(f"MIDI output: {ports[MIDI_OUTPUT_PORT]}")
        self._midi_out.open_port(MIDI_OUTPUT_PORT)

    def stop(self):
        self._midi_out.close_port()

    def play_phrase(
        self,
        pitches:   list[int],
        durations: list[float],          # per-note duration in seconds
        velocity   = DEFAULT_VELOCITY,   # int OR list[int] for per-note dynamics
        channel:   int   = 1,
    ):
        """
        Play a phrase note-by-note with per-note durations.

        velocity may be a single int (applied to all notes) or a list of ints
        (one per note) for per-note dynamic shaping from the energy arc.

        Each note sounds for ARTICULATION_RATIO of its slot duration, with a
        short silence before the next note — giving natural sax articulation.
        """
        ch = channel - 1
        for i, (pitch, dur) in enumerate(zip(pitches, durations)):
            vel         = velocity[i] if isinstance(velocity, list) else velocity
            vel         = max(1, min(127, int(vel)))
            sound_dur   = max(0.02, dur * ARTICULATION_RATIO)
            silence_dur = max(0.005, dur - sound_dur)
            self._midi_out.send_message([0x90 | ch, pitch, vel])
            time.sleep(sound_dur)
            self._midi_out.send_message([0x80 | ch, pitch, 0])
            time.sleep(silence_dur)

    def silence(self, channel: int = 1):
        """Send all-notes-off on the output channel."""
        ch = channel - 1
        self._midi_out.send_message([0xB0 | ch, 123, 0])

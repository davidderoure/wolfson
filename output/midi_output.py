"""Sends generated sax phrases to a MIDI output port."""

import rtmidi
import time
from config import MIDI_OUTPUT_PORT

DEFAULT_VELOCITY = 80
DEFAULT_NOTE_DURATION = 0.25   # seconds; will be made dynamic later


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

    def play_phrase(self, pitches, duration=DEFAULT_NOTE_DURATION, velocity=DEFAULT_VELOCITY, channel=1):
        """Play a list of MIDI pitches as a sequence of notes."""
        ch = channel - 1   # rtmidi uses 0-indexed channels
        for pitch in pitches:
            self._midi_out.send_message([0x90 | ch, pitch, velocity])
            time.sleep(duration)
            self._midi_out.send_message([0x80 | ch, pitch, 0])

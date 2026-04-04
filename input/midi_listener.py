"""Real-time MIDI input from the bass pitch-to-MIDI interface."""

import rtmidi
import threading
import time
from config import MIDI_INPUT_PORT, MIDI_PITCH_MIN, MIDI_PITCH_MAX


class MidiListener:
    """Listens on a MIDI input port and dispatches note events."""

    def __init__(self, on_note_on, on_note_off):
        self.on_note_on = on_note_on
        self.on_note_off = on_note_off
        self._midi_in = rtmidi.MidiIn()

    def start(self):
        ports = self._midi_in.get_ports()
        if not ports:
            raise RuntimeError("No MIDI input ports found.")
        print(f"MIDI input: {ports[MIDI_INPUT_PORT]}")
        self._midi_in.open_port(MIDI_INPUT_PORT)
        self._midi_in.set_callback(self._callback)

    def stop(self):
        self._midi_in.close_port()

    def _callback(self, event, _data):
        message, _delta = event
        status = message[0] & 0xF0
        pitch = message[1]
        velocity = message[2]
        if not (MIDI_PITCH_MIN <= pitch <= MIDI_PITCH_MAX):
            return
        if status == 0x90 and velocity > 0:
            self.on_note_on(pitch, velocity, time.time())
        elif status == 0x80 or (status == 0x90 and velocity == 0):
            self.on_note_off(pitch, time.time())

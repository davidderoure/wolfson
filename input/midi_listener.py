"""Real-time MIDI input from the bass pitch-to-MIDI interface."""

import rtmidi
import threading
import time
from config import MIDI_INPUT_PORT, MIDI_PITCH_MIN, MIDI_PITCH_MAX, MIDI_VELOCITY_MIN


class MidiListener:
    """Listens on a MIDI input port and dispatches note events."""

    def __init__(self, on_note_on, on_note_off):
        self.on_note_on = on_note_on
        self.on_note_off = on_note_off
        self._midi_in = rtmidi.MidiIn()

    @staticmethod
    def check_ports():
        """
        Validate the configured MIDI input port before the UI starts.
        Raises SystemExit with a readable message if the port is missing.
        Call this before dashboard.start() so the error is always visible.
        """
        ports = rtmidi.MidiIn().get_ports()
        if not ports:
            raise SystemExit(
                "Error: no MIDI input ports found.\n"
                "Check that your MIDI interface is connected and recognised by the OS."
            )
        if MIDI_INPUT_PORT >= len(ports):
            raise SystemExit(
                f"Error: MIDI_INPUT_PORT={MIDI_INPUT_PORT} is out of range "
                f"({len(ports)} port{'s' if len(ports) != 1 else ''} available).\n"
                f"Available ports: {ports}\n"
                f"Update MIDI_INPUT_PORT in config.py."
            )

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
            if velocity < MIDI_VELOCITY_MIN:
                return
            self.on_note_on(pitch, velocity, time.time())
        elif status == 0x80 or (status == 0x90 and velocity == 0):
            self.on_note_off(pitch, time.time())

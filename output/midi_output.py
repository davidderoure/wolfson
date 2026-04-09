"""Sends generated sax phrases to a MIDI output port."""

import rtmidi
import time
import threading
from config import MIDI_OUTPUT_PORT, REST_PITCH

# Semitone intervals from root for each chord quality (maj7, dom7, min7, dim7)
_CHORD_VOICINGS = {
    0: (0, 4, 7, 11),   # major  →  R  3  5  maj7
    1: (0, 4, 7, 10),   # dom    →  R  3  5  b7
    2: (0, 3, 7, 10),   # minor  →  R  b3 5  b7
    3: (0, 3, 6,  9),   # dim    →  R  b3 b5 bb7
}

DEFAULT_VELOCITY    = 80
ARTICULATION_RATIO  = 0.85   # note sounds for this fraction of its slot; rest is silence
MAX_NOTE_DUR        = 5.0    # hard cap (seconds) — prevents runaway held notes


class MidiOutput:
    """
    Thread-safe MIDI output with a single dedicated playback thread.

    play_phrase() is non-blocking and "latest wins": submitting a new phrase
    while one is playing causes the output thread to finish its current note,
    send note_off, then switch.  Because only one thread ever writes note
    events to the MIDI device, concurrent-phrase races and stuck notes are
    architecturally impossible rather than guarded against by locks.

    Chord hints (play_chord_hint) run on their own short-lived daemon thread
    on a separate MIDI channel, so they never contend with phrase playback.
    """

    def __init__(self):
        self._midi_out = rtmidi.MidiOut()

        # Single-slot phrase queue — protected by _pending_lock.
        # play_phrase() writes; only the output thread reads and clears.
        # "Latest wins": a new phrase overwrites any phrase not yet started.
        self._pending      = None            # (pitches, durations, velocity, channel, on_complete)
        self._pending_lock = threading.Lock()

        self._wake       = threading.Event()   # wakes output thread when phrase arrives
        self._shutdown   = threading.Event()   # tells output thread to exit
        self._is_playing = False               # True while output thread is in _play_blocking

        self._output_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self):
        ports = self._midi_out.get_ports()
        if not ports:
            raise RuntimeError("No MIDI output ports found.")
        print(f"MIDI output: {ports[MIDI_OUTPUT_PORT]}")
        self._midi_out.open_port(MIDI_OUTPUT_PORT)
        self._shutdown.clear()
        self._output_thread = threading.Thread(
            target=self._run, name="midi-output", daemon=True,
        )
        self._output_thread.start()

    def stop(self, silence_channels=None):
        """
        Stop the output thread, optionally silence MIDI channels, close port.

        Joins the output thread before sending any silence messages so the
        thread and the caller are never writing to the device concurrently.

        silence_channels — int or list[int] of 1-indexed MIDI channels to
                           silence before closing (same semantics as silence()).
        """
        self._shutdown.set()
        self._wake.set()          # unblock the thread if it is waiting
        if self._output_thread and self._output_thread.is_alive():
            self._output_thread.join(timeout=2.0)
        # Output thread has exited — we now own the device exclusively.
        if silence_channels is not None:
            self._send_silence(silence_channels)
        self._midi_out.close_port()

    # ------------------------------------------------------------------
    # Public playback interface
    # ------------------------------------------------------------------

    @property
    def is_playing(self) -> bool:
        """True while the output thread is actively playing a phrase."""
        return self._is_playing

    def play_phrase(
        self,
        pitches:    list[int],
        durations:  list[float],          # per-note duration in seconds
        velocity    = DEFAULT_VELOCITY,   # int OR list[int] for per-note dynamics
        channel:    int = 1,
        on_complete = None,               # optional callable; fired when phrase ends
    ):
        """
        Submit a phrase for playback.  Returns immediately (non-blocking).

        The output thread plays notes one at a time.  If a phrase is already
        playing, the thread finishes its current note (including note_off),
        then switches to this phrase.  Any previously queued but unstarted
        phrase is discarded — latest call always wins.

        velocity may be a single int or a list of ints (one per note).

        on_complete — if provided, called by the output thread after this
        phrase ends (whether naturally or superseded by a newer phrase).
        Use this to defer arc-controller bookkeeping until playback actually
        finishes rather than firing immediately on submission.
        """
        with self._pending_lock:
            self._pending = (pitches, durations, velocity, channel, on_complete)
        self._wake.set()

    def play_chord_hint(
        self,
        chord_idx:    int,
        beat_dur_sec: float,
        channel:      int = 3,
        velocity:     int = 55,
    ):
        """
        Play a voiced 4-note chord on the hint channel in a background thread.

        Fired once per bass phrase to make the internal harmonic state
        directly audible — like a pianist lightly picking out the chord at
        the start of each exchange.  The chord sounds for 1.5 beats then
        releases silently.

        chord_idx    — Wolfson chord index (0–47); NC_INDEX (48) = silent
        beat_dur_sec — seconds per beat at the current tempo
        channel      — 1-indexed MIDI channel (default 3)
        velocity     — MIDI velocity; 55 is a quiet comp
        """
        from data.chords import NC_INDEX, N_QUALITIES
        if chord_idx == NC_INDEX:
            return

        root_pc   = chord_idx // N_QUALITIES   # 0–11
        quality   = chord_idx %  N_QUALITIES   # 0–3
        root_midi = 48 + root_pc               # C3–B3
        notes     = [root_midi + i for i in _CHORD_VOICINGS.get(quality, _CHORD_VOICINGS[0])]
        dur       = beat_dur_sec * 1.5
        ch        = channel - 1

        def _play():
            for n in notes:
                self._midi_out.send_message([0x90 | ch, n, velocity])
            time.sleep(dur)
            for n in notes:
                self._midi_out.send_message([0x80 | ch, n, 0])

        threading.Thread(target=_play, daemon=True).start()

    def silence(self, channels=1):
        """
        Silence one or more MIDI channels immediately.

        Sends CC 123 (All Notes Off) and CC 120 (All Sound Off) followed by
        explicit note_off for every pitch — Logic software instruments ignore
        the CC messages and need per-pitch note_offs.

        Safe to call when the output thread is idle (e.g. at session start).
        At shutdown, use stop(silence_channels=...) instead so the thread is
        joined before silence messages are sent.

        channels — int or list/tuple of 1-indexed MIDI channel numbers.
        """
        self._send_silence(channels)

    # ------------------------------------------------------------------
    # Internal helpers — called only from output thread (except _send_silence)
    # ------------------------------------------------------------------

    def _send_silence(self, channels):
        if isinstance(channels, int):
            channels = [channels]
        for channel in channels:
            ch = channel - 1
            self._midi_out.send_message([0xB0 | ch, 123, 0])   # All Notes Off
            self._midi_out.send_message([0xB0 | ch, 120, 0])   # All Sound Off
            for pitch in range(128):
                self._midi_out.send_message([0x80 | ch, pitch, 0])

    def _flush_channel(self, ch: int):
        """Send note_off for every pitch on a 0-indexed channel."""
        for pitch in range(128):
            self._midi_out.send_message([0x80 | ch, pitch, 0])

    def _run(self):
        """Output thread main loop: plays one phrase at a time, latest wins."""
        while not self._shutdown.is_set():
            self._wake.wait(timeout=0.1)
            self._wake.clear()
            if self._shutdown.is_set():
                return

            with self._pending_lock:
                item          = self._pending
                self._pending = None

            if item is None:
                continue

            pitches, durations, velocity, channel, on_complete = item
            self._is_playing = True
            self._play_blocking(pitches, durations, velocity, channel)
            self._is_playing = False
            # Notify caller that this phrase has ended (naturally or superseded).
            if on_complete is not None and not self._shutdown.is_set():
                on_complete()

    def _play_blocking(self, pitches, durations, velocity, channel):
        """
        Play a phrase note-by-note.  Called only from the output thread.

        Design guarantees:
        • note_off is sent after every note_on unconditionally — no pitch is
          ever left sounding regardless of what happens next.
        • Between notes the thread checks _pending: if a newer phrase has
          arrived it exits early, and _run immediately picks up that phrase.
        • The flush at entry clears any residue from the previous phrase.
        """
        ch = channel - 1
        self._flush_channel(ch)

        for i, (pitch, dur) in enumerate(zip(pitches, durations)):

            # Check for a superseding phrase or shutdown before each note.
            if self._shutdown.is_set():
                return
            with self._pending_lock:
                if self._pending is not None:
                    return

            if pitch == REST_PITCH:
                time.sleep(dur)
                continue

            vel         = velocity[i] if isinstance(velocity, list) else velocity
            vel         = max(1, min(127, int(vel)))
            dur         = min(dur, MAX_NOTE_DUR)
            sound_dur   = max(0.02, dur * ARTICULATION_RATIO)
            silence_dur = max(0.005, dur - sound_dur)

            self._midi_out.send_message([0x90 | ch, pitch, vel])
            time.sleep(sound_dur)
            self._midi_out.send_message([0x80 | ch, pitch, 0])   # unconditional

            # Exit before the inter-note gap if superseded — the new phrase
            # will flush the channel when it starts.
            if self._shutdown.is_set():
                return
            with self._pending_lock:
                if self._pending is not None:
                    return

            time.sleep(silence_dur)

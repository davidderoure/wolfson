"""Sends generated sax phrases to a MIDI output port."""

import rtmidi
import time
import threading
from config import MIDI_OUTPUT_PORT, REST_PITCH

# Semitone intervals from root for each chord quality (maj7, dom7, min7, dim7).
# Used in tonal/progression sections where chord identity matters.
_CHORD_VOICINGS = {
    0: (0, 4, 7, 11),   # major  →  R  3  5  maj7
    1: (0, 4, 7, 10),   # dom    →  R  3  5  b7
    2: (0, 3, 7, 10),   # minor  →  R  b3 5  b7
    3: (0, 3, 6,  9),   # dim    →  R  b3 b5 bb7
}

# Quartal voicing: four stacked perfect 4ths from the root (R P4 m7 P11).
# Quality-independent — the same intervals work for any root in a modal
# context because the goal is harmonic ambiguity rather than functional
# clarity.  Used during the modal harmonic mode (building / recapitulation)
# to match the floating, non-resolving character of Dorian / Phrygian etc.
# Compare: Dm7 = D F A C (pulls toward G or F); D quartal = D G C F
# (no minor or major 3rd — sounds modal rather than functional).
_QUARTAL_VOICING = (0, 5, 10, 15)   # R, P4, m7, P11

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

        # Chord hint cancellation — generation counter per MIDI channel (0-indexed).
        # Each play_chord_hint() call increments the counter for that channel and
        # silences any notes left over from the previous chord before starting the
        # new one.  The old daemon thread checks on wake-up whether its generation
        # is still current; if not, it skips its note_offs so they can't cut the
        # new chord short.
        self._chord_lock  = threading.Lock()
        self._chord_gen   = {}   # ch (0-indexed) -> int generation counter
        self._chord_notes = {}   # ch (0-indexed) -> list[int] currently sounding notes

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @staticmethod
    def check_ports():
        """
        Validate the configured MIDI output port before the UI starts.
        Raises SystemExit with a readable message if the port is missing.
        Call this before dashboard.start() so the error is always visible.
        """
        ports = rtmidi.MidiOut().get_ports()
        if not ports:
            raise SystemExit(
                "Error: no MIDI output ports found.\n"
                "Check that your MIDI interface is connected and recognised by the OS."
            )
        if MIDI_OUTPUT_PORT >= len(ports):
            raise SystemExit(
                f"Error: MIDI_OUTPUT_PORT={MIDI_OUTPUT_PORT} is out of range "
                f"({len(ports)} port{'s' if len(ports) != 1 else ''} available).\n"
                f"Available ports: {ports}\n"
                f"Update MIDI_OUTPUT_PORT in config.py."
            )

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
        channel:      int   = 3,
        velocity:     int   = 70,
        dur_beats:    float = 1.5,
        quartal:      bool  = False,
    ):
        """
        Play a voiced 4-note chord on the hint channel in a background thread.

        Fired once per bass phrase to make the internal harmonic state
        directly audible — like a pianist lightly picking out the chord at
        the start of each exchange.

        If a chord is already sounding on this channel when a new one is
        requested, the old chord is silenced immediately and its daemon thread
        skips its own note_offs (generation counter mismatch), preventing any
        overlap or premature cutoff regardless of dur_beats.

        chord_idx    — Wolfson chord index (0–47); NC_INDEX (48) = silent
        beat_dur_sec — seconds per beat at the current tempo
        channel      — 1-indexed MIDI channel (default 3)
        velocity     — MIDI velocity; 55 is a quiet comp
        dur_beats    — how long the chord sounds in beats (default 1.5)
                       increase for piano sustain; use full phrase length
                       for pads — cancellation keeps this safe regardless
        quartal      — if True, voice as four stacked perfect 4ths regardless
                       of chord quality; suited to modal sections where
                       ambiguity is preferred over functional identity
        """
        from data.chords import NC_INDEX, N_QUALITIES
        if chord_idx == NC_INDEX:
            return

        root_pc   = chord_idx // N_QUALITIES   # 0–11
        quality   = chord_idx %  N_QUALITIES   # 0–3
        root_midi = 48 + root_pc               # C3–B3
        intervals = _QUARTAL_VOICING if quartal else _CHORD_VOICINGS.get(quality, _CHORD_VOICINGS[0])
        notes     = [root_midi + i for i in intervals]
        dur       = beat_dur_sec * dur_beats
        ch        = channel - 1

        # Acquire generation for this channel, silence previous chord.
        with self._chord_lock:
            gen = self._chord_gen.get(ch, 0) + 1
            self._chord_gen[ch]   = gen
            old_notes = self._chord_notes.get(ch, [])
            for n in old_notes:
                self._midi_out.send_message([0x80 | ch, n, 0])
            self._chord_notes[ch] = notes

        # Send note_ons outside the lock (non-blocking MIDI write).
        for n in notes:
            self._midi_out.send_message([0x90 | ch, n, velocity])

        def _release(my_gen):
            time.sleep(dur)
            with self._chord_lock:
                if self._chord_gen.get(ch) != my_gen:
                    return   # superseded — new chord already took over
                for n in notes:
                    self._midi_out.send_message([0x80 | ch, n, 0])
                self._chord_notes[ch] = []

        threading.Thread(target=_release, args=(gen,), daemon=True).start()

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

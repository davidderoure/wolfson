"""Segments the incoming note stream into discrete bass phrases."""

import time
import threading
from config import SILENCE_THRESHOLD_SEC, MIN_PHRASE_NOTES


class PhraseDetector:
    """
    Accumulates notes and fires on_phrase_complete when a silence gap
    exceeds SILENCE_THRESHOLD_SEC.

    A phrase is a list of dicts: {pitch, velocity, onset, offset}
    """

    def __init__(self, on_phrase_complete):
        self.on_phrase_complete = on_phrase_complete
        self._current_phrase = []
        self._active_notes = {}   # pitch -> onset time
        self._timer = None
        self._lock = threading.Lock()

    def note_on(self, pitch, velocity, t):
        with self._lock:
            self._cancel_timer()
            # Enforce monophony: force-close any currently active notes so
            # that a held note doesn't prevent the silence timer from firing.
            # This matches the behaviour of a monophonic bass instrument and
            # means legato playing is handled correctly (previous note ends
            # when the next begins).  A MIDI keyboard playing a true chord
            # produces near-zero-duration grace notes, which is harmless.
            for active_pitch, (onset, vel) in list(self._active_notes.items()):
                self._current_phrase.append({
                    "pitch":    active_pitch,
                    "velocity": vel,
                    "onset":    onset,
                    "offset":   t,
                })
            self._active_notes.clear()
            self._active_notes[pitch] = (t, velocity)

    def note_off(self, pitch, t):
        with self._lock:
            if pitch in self._active_notes:
                onset, velocity = self._active_notes.pop(pitch)
                self._current_phrase.append({
                    "pitch": pitch,
                    "velocity": velocity,
                    "onset": onset,
                    "offset": t,
                })
            if not self._active_notes:
                self._start_timer()

    def _start_timer(self):
        self._timer = threading.Timer(SILENCE_THRESHOLD_SEC, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    def _flush(self):
        with self._lock:
            phrase = self._current_phrase
            self._current_phrase = []
        if len(phrase) >= MIN_PHRASE_NOTES:
            self.on_phrase_complete(phrase)

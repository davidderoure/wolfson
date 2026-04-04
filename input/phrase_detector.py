"""Segments the incoming note stream into discrete bass phrases."""

import time
import threading
from config import SILENCE_THRESHOLD_SEC, MIN_PHRASE_NOTES, MIDI_MIN_NOTE_DUR


class PhraseDetector:
    """
    Accumulates notes and fires on_phrase_complete when a silence gap
    exceeds SILENCE_THRESHOLD_SEC.

    A phrase is a list of dicts: {pitch, velocity, onset, offset}

    A watchdog timer fires if a note_off is never received for the last note
    (e.g. dropped by the pitch-to-MIDI converter), so the phrase is never
    left stranded in an un-flushed state.
    """

    def __init__(self, on_phrase_complete, silence_threshold=None, min_note_dur=None):
        self.on_phrase_complete  = on_phrase_complete
        self._silence_threshold  = silence_threshold if silence_threshold is not None \
                                   else SILENCE_THRESHOLD_SEC
        self._min_note_dur       = min_note_dur if min_note_dur is not None \
                                   else MIDI_MIN_NOTE_DUR
        self._current_phrase = []
        self._active_notes = {}   # pitch -> (onset, velocity)
        self._timer    = None
        self._watchdog = None
        self._lock = threading.Lock()

    def note_on(self, pitch, velocity, t):
        with self._lock:
            self._cancel_timer()
            self._cancel_watchdog()
            # Enforce monophony: force-close any currently active notes so
            # that a held note doesn't prevent the silence timer from firing.
            # This matches the behaviour of a monophonic bass instrument and
            # means legato playing is handled correctly (previous note ends
            # when the next begins).  A MIDI keyboard playing a true chord
            # produces near-zero-duration grace notes, which is harmless.
            for active_pitch, (onset, vel) in list(self._active_notes.items()):
                if t - onset >= self._min_note_dur:
                    self._current_phrase.append({
                        "pitch":    active_pitch,
                        "velocity": vel,
                        "onset":    onset,
                        "offset":   t,
                    })
            self._active_notes.clear()
            self._active_notes[pitch] = (t, velocity)
            # Watchdog: if no note_off arrives within the silence threshold,
            # the pitch-to-MIDI converter has dropped it.  Force-close the
            # note and flush the phrase so it isn't left stranded.
            self._start_watchdog(pitch)

    def note_off(self, pitch, t):
        with self._lock:
            self._cancel_watchdog()
            if pitch in self._active_notes:
                onset, velocity = self._active_notes.pop(pitch)
                if t - onset >= self._min_note_dur:
                    self._current_phrase.append({
                        "pitch": pitch,
                        "velocity": velocity,
                        "onset": onset,
                        "offset": t,
                    })
            if not self._active_notes:
                self._start_timer()

    # --- Silence timer (fires after note_off) ---

    def _start_timer(self):
        self._timer = threading.Timer(self._silence_threshold, self._flush)
        self._timer.daemon = True
        self._timer.start()

    def _cancel_timer(self):
        if self._timer:
            self._timer.cancel()
            self._timer = None

    # --- Watchdog timer (fires if note_off never arrives) ---

    def _start_watchdog(self, pitch):
        self._watchdog = threading.Timer(
            self._silence_threshold, self._watchdog_fire, args=(pitch,)
        )
        self._watchdog.daemon = True
        self._watchdog.start()

    def _cancel_watchdog(self):
        if self._watchdog:
            self._watchdog.cancel()
            self._watchdog = None

    def _watchdog_fire(self, pitch):
        """Force-close a stuck note then start the silence timer."""
        t = time.time()
        with self._lock:
            if pitch in self._active_notes:
                onset, vel = self._active_notes.pop(pitch)
                self._current_phrase.append({
                    "pitch":    pitch,
                    "velocity": vel,
                    "onset":    onset,
                    "offset":   t,
                })
            if not self._active_notes:
                self._start_timer()

    # --- Flush ---

    def _flush(self):
        with self._lock:
            phrase = self._current_phrase
            self._current_phrase = []
        if len(phrase) >= MIN_PHRASE_NOTES:
            self.on_phrase_complete(phrase)

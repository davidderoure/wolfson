"""
Unit tests for PhraseDetector state-machine logic.

No MIDI hardware required — all note events are injected directly via
note_on() / note_off() with synthetic timestamps.  A very short
silence_threshold (0.05 s) is used so tests that need the timer to fire
can do so quickly.

Tests cover:
  - Normal phrase completion (silence timer after last note_off)
  - Ghost notes (duration < min_note_dur) do NOT start the silence timer
  - Stale note_offs (pitch not in active_notes) do NOT cancel the watchdog
  - Watchdog fires when note is held past silence_threshold
  - Monophony: new note_on while a note is active closes the previous note
  - Minimum phrase length: single-note phrase fires normally

Run with:
    cd ~/wolfson
    python -m pytest tests/test_phrase_detector.py -v
or:
    python tests/test_phrase_detector.py
"""

import sys
import time
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from input.phrase_detector import PhraseDetector
from input.midi_listener import MidiListener


# Short threshold so tests don't have to wait multiple seconds.
THRESH = 0.06   # 60 ms silence = phrase end
MIN_DUR = 0.05  # minimum note duration to be kept (matches config default)


def make_detector(on_phrase=None, silence_threshold=THRESH, min_note_dur=MIN_DUR,
                  min_phrase_notes=1):
    phrases = []
    if on_phrase is None:
        def on_phrase(p):
            phrases.append(p)
    return PhraseDetector(
        on_phrase_complete=on_phrase,
        silence_threshold=silence_threshold,
        min_note_dur=min_note_dur,
        min_phrase_notes=min_phrase_notes,
    ), phrases


# ---------------------------------------------------------------------------
# Helper: wait up to `timeout` seconds for a condition, polling at 5 ms.
# ---------------------------------------------------------------------------

def wait_for(condition, timeout=0.5):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(0.005)
    return False


# ---------------------------------------------------------------------------
# Basic phrase completion
# ---------------------------------------------------------------------------

class TestBasicPhrase(unittest.TestCase):

    def test_single_note_phrase(self):
        """A single note followed by silence should produce a one-note phrase."""
        det, phrases = make_detector()
        t = time.time()
        det.note_on(60, 80, t)
        det.note_off(60, t + 0.10)   # note duration 100 ms > MIN_DUR
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase callback did not fire for single-note phrase")
        self.assertEqual(len(phrases[0]), 1)
        self.assertEqual(phrases[0][0]["pitch"], 60)

    def test_multi_note_phrase(self):
        """Several notes in quick succession should be grouped into one phrase."""
        det, phrases = make_detector()
        t = time.time()
        # Durations must exceed MIN_DUR (50 ms) or they are discarded as ghosts.
        # Use 60 ms on, 10 ms off between notes.
        for i, pitch in enumerate([60, 62, 64, 65]):
            det.note_on(pitch, 80,  t + i * 0.07)
            det.note_off(pitch,     t + i * 0.07 + 0.06)
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase callback did not fire")
        self.assertEqual(len(phrases[0]), 4)

    def test_two_separate_phrases(self):
        """Two note groups separated by > silence_threshold produce two phrases."""
        det, phrases = make_detector()
        t = time.time()
        # First phrase
        det.note_on(60, 80, t)
        det.note_off(60, t + 0.10)
        # Wait for first phrase to fire
        wait_for(lambda: len(phrases) == 1, timeout=0.5)
        # Second phrase (after a gap)
        t2 = time.time()
        det.note_on(62, 80, t2)
        det.note_off(62, t2 + 0.10)
        fired = wait_for(lambda: len(phrases) == 2, timeout=0.5)
        self.assertTrue(fired, "Second phrase did not fire")


# ---------------------------------------------------------------------------
# Ghost note (duration < min_note_dur)
# ---------------------------------------------------------------------------

class TestGhostNote(unittest.TestCase):

    def test_ghost_note_alone_does_not_fire(self):
        """A ghost note (too short) on its own should not start the silence timer."""
        det, phrases = make_detector()
        t = time.time()
        # Duration = 5 ms, well below MIN_DUR = 50 ms
        det.note_on(60, 80, t)
        det.note_off(60, t + 0.005)
        # Wait long enough for the timer to have fired if it were started
        time.sleep(THRESH * 3)
        self.assertEqual(len(phrases), 0,
                         "Silence timer should not fire after a ghost note alone")

    def test_ghost_note_mid_phrase_does_not_cause_premature_end(self):
        """
        A ghost note arriving while playing should not start a new silence timer
        that expires before the next real note.

        Scenario:
          - Real note A played and released
          - Ghost note (same tick) arrives immediately after
          - Small pause (< silence_threshold) before real note B
          - Real note B played — all should be in ONE phrase
        """
        det, phrases = make_detector()
        t = time.time()
        # Real note A
        det.note_on(60, 80, t)
        det.note_off(60, t + 0.10)
        # Ghost note immediately after (1 ms duration)
        det.note_on(61, 80, t + 0.10)
        det.note_off(61, t + 0.101)
        # Pause of 0.04 s (< THRESH=0.06) then real note B
        det.note_on(62, 80, t + 0.14)
        det.note_off(62, t + 0.24)

        # Wait for the phrase to fire
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Single phrase should fire after note B")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches, "Note A should be in the phrase")
        self.assertIn(62, pitches, "Note B should be in the phrase")
        self.assertNotIn(61, pitches, "Ghost note pitch should be dropped")

    def test_ghost_note_does_not_split_phrase_before_silence(self):
        """
        Before our fix, a ghost note arriving after a real note would start the
        silence timer, causing a premature phrase split if the next real note
        took ~silence_threshold seconds to arrive.  The phrase should remain
        intact as long as the gap is actually shorter than silence_threshold.
        """
        det, phrases = make_detector()
        t = time.time()

        # Real note played and released
        det.note_on(65, 80, t)
        det.note_off(65, t + 0.12)

        # Ghost arrives immediately
        det.note_on(66, 80, t + 0.12)
        det.note_off(66, t + 0.121)

        # Next real note arrives 0.04 s later — still within THRESH=0.06 from
        # the real note_off, but 0.04+0.001 = 0.041 s from the ghost note_off.
        # Before the fix the ghost would have started the 0.06 s timer and
        # the phrase would have fired before this next note arrived.
        time.sleep(0.04)
        det.note_on(67, 80, time.time())
        det.note_off(67, time.time() + 0.12)

        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase should fire once as a single unit")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(65, pitches)
        self.assertIn(67, pitches)
        self.assertNotIn(66, pitches)   # ghost dropped
        self.assertEqual(len(phrases), 1, "Should not have split into two phrases")


# ---------------------------------------------------------------------------
# Stale note_off (i2M monophony behaviour)
# ---------------------------------------------------------------------------

class TestStaleNoteOff(unittest.TestCase):

    def test_stale_note_off_does_not_affect_active_phrase(self):
        """
        A note_off for a pitch not currently in active_notes (stale) should
        not start the silence timer or disturb the current active note.
        """
        det, phrases = make_detector()
        t = time.time()

        # Note A starts
        det.note_on(60, 80, t)
        # Note B starts while A is still held (monophony closes A internally)
        det.note_on(62, 80, t + 0.01)
        # Now a stale note_off for A arrives (i2M sends this late)
        det.note_off(60, t + 0.03)   # pitch 60 is no longer in active_notes
        # Note B ends normally
        det.note_off(62, t + 0.10)

        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase should still fire after stale note_off")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(62, pitches)

    def test_stale_note_off_does_not_cancel_watchdog(self):
        """
        If the watchdog is set for the current note and a stale note_off for
        a different pitch arrives, the watchdog must NOT be cancelled.
        The current note should still produce a phrase via the watchdog.
        """
        # Use a longer threshold so we can observe whether the watchdog fires.
        # silence_threshold=0.15 s, stale note_off at 0.05 s.
        det, phrases = make_detector(silence_threshold=0.15)
        t = time.time()

        # Note A starts
        det.note_on(60, 80, t)
        # Note B starts while A held — monophony closes A
        det.note_on(62, 80, t + 0.01)
        # Stale note_off for A (should be ignored)
        time.sleep(0.05)
        det.note_off(60, time.time())
        # B should still be open; wait for watchdog
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired,
                        "Watchdog should fire for note B despite stale note_off for A")


# ---------------------------------------------------------------------------
# Watchdog — note held past silence_threshold
# ---------------------------------------------------------------------------

class TestWatchdog(unittest.TestCase):

    def test_watchdog_fires_for_sustained_note(self):
        """
        When a note is held longer than silence_threshold with no note_off,
        the watchdog should end the phrase.
        """
        det, phrases = make_detector(silence_threshold=0.06)
        t = time.time()
        det.note_on(60, 80, t)
        # Do NOT send note_off — watchdog should fire after ~0.06 s
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Watchdog did not fire for sustained note")
        self.assertEqual(phrases[0][0]["pitch"], 60)


# ---------------------------------------------------------------------------
# Monophony
# ---------------------------------------------------------------------------

class TestMonophony(unittest.TestCase):

    def test_new_note_closes_previous(self):
        """Starting a second note while the first is held should close the first."""
        det, phrases = make_detector()
        t = time.time()
        det.note_on(60, 80, t)
        # Hold note 60 for 70 ms (> MIN_DUR=50 ms) before note 62 arrives,
        # otherwise the monophony handler discards it as too short.
        time.sleep(0.07)
        det.note_on(62, 80, t + 0.07)   # closes 60 implicitly
        det.note_off(62, t + 0.17)
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired)
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches)
        self.assertIn(62, pitches)


# ---------------------------------------------------------------------------
# Notes shorter than min_note_dur are filtered
# ---------------------------------------------------------------------------

class TestMinNoteDur(unittest.TestCase):

    def test_short_notes_filtered_from_phrase(self):
        """Notes below min_note_dur should not appear in the phrase."""
        det, phrases = make_detector(min_note_dur=0.05)
        t = time.time()
        # Real note
        det.note_on(60, 80, t)
        det.note_off(60, t + 0.10)
        # Too-short note
        det.note_on(61, 80, t + 0.11)
        det.note_off(61, t + 0.112)   # 2 ms — below 50 ms floor
        # Another real note
        det.note_on(62, 80, t + 0.12)
        det.note_off(62, t + 0.22)
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired)
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches)
        self.assertIn(62, pitches)
        self.assertNotIn(61, pitches, "Sub-minimum note should be filtered out")


# ---------------------------------------------------------------------------
# Scenario 1 — i2M note extend: note_off arrives after next note_on
# ---------------------------------------------------------------------------

class TestScenario1_NoteExtend(unittest.TestCase):
    """
    i2M 'note extend' hardware behaviour: the interface holds a note_off until
    the next note_on arrives.  This means every note_off for note N is stale —
    it arrives after note N+1 has already started and been closed by monophony.

    Pattern for a three-note run:
        note_on(A)
        note_on(B)          ← monophony closes A
        note_off(A)         ← stale (A no longer in active_notes)
        note_on(C)          ← monophony closes B
        note_off(B)         ← stale
        note_off(C)         ← real, starts silence timer
    """

    def test_all_notes_in_one_phrase(self):
        """All notes should appear in a single phrase despite stale note_offs."""
        det, phrases = make_detector()
        gap = 0.08   # 80 ms between note_ons — well above MIN_DUR = 50 ms

        t = time.time()
        det.note_on(60, 80, t)
        det.note_on(62, 80, t + gap)            # monophony closes 60
        det.note_off(60, t + gap + 0.01)        # stale — must be ignored
        det.note_on(64, 80, t + gap * 2)        # monophony closes 62
        det.note_off(62, t + gap * 2 + 0.01)   # stale — must be ignored
        det.note_off(64, t + gap * 3)           # real — starts silence timer

        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase should fire after final real note_off")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches, "Note A (60) should be in phrase")
        self.assertIn(62, pitches, "Note B (62) should be in phrase")
        self.assertIn(64, pitches, "Note C (64) should be in phrase")
        self.assertEqual(len(phrases), 1, "Must not split into multiple phrases")

    def test_stale_note_off_does_not_start_premature_timer(self):
        """
        A stale note_off must not start the silence timer while the next
        real note is still active, even if _current_phrase already has content.
        """
        det, phrases = make_detector()
        gap = 0.08

        t = time.time()
        det.note_on(60, 80, t)
        det.note_on(62, 80, t + gap)         # monophony closes 60
        det.note_off(60, t + gap + 0.01)     # stale — must NOT start timer

        # Verify: no phrase fires during the silence window (note 62 still active)
        time.sleep(THRESH * 2)
        self.assertEqual(len(phrases), 0,
                         "Stale note_off must not trigger a premature phrase end")

        # Now close note 62 properly and confirm the phrase does fire
        det.note_off(62, time.time())
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Phrase should fire after real note_off for note B")
        self.assertIn(62, [n["pitch"] for n in phrases[0]])


# ---------------------------------------------------------------------------
# Scenario 2 — note_off completely missing for every note in a run
# ---------------------------------------------------------------------------

class TestScenario2_MissingNoteOffs(unittest.TestCase):
    """
    Hardware or cable dropout causes every note_off to be lost.
    Monophony in note_on() closes each earlier note when the next starts;
    the watchdog timer closes the final held note so the phrase is never
    left stranded.
    """

    def test_run_with_no_note_offs(self):
        """
        A run of note_ons with no note_offs at all should still produce a
        complete phrase: monophony covers the first N-1 notes, the watchdog
        covers the last.
        """
        det, phrases = make_detector(silence_threshold=THRESH)
        gap = 0.08   # 80 ms between note_ons — above MIN_DUR = 50 ms

        det.note_on(60, 80, time.time())
        time.sleep(gap)
        det.note_on(62, 80, time.time())   # monophony closes 60
        time.sleep(gap)
        det.note_on(64, 80, time.time())   # monophony closes 62
        # No note_off for 64 — watchdog fires after THRESH

        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Watchdog should fire for the last held note")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches, "Note 60 (closed by monophony) should be in phrase")
        self.assertIn(62, pitches, "Note 62 (closed by monophony) should be in phrase")
        self.assertIn(64, pitches, "Note 64 (closed by watchdog) should be in phrase")
        self.assertEqual(len(phrases), 1, "Must be a single phrase")

    def test_single_note_no_note_off(self):
        """
        A single note with no note_off should produce a one-note phrase
        via the watchdog, with no further phrases afterwards.
        """
        det, phrases = make_detector(silence_threshold=THRESH)
        det.note_on(60, 80, time.time())
        # No note_off — watchdog fires after THRESH
        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired, "Watchdog should fire for single held note")
        self.assertEqual(len(phrases[0]), 1)
        self.assertEqual(phrases[0][0]["pitch"], 60)
        # No second phrase should appear
        time.sleep(THRESH * 2)
        self.assertEqual(len(phrases), 1, "No spurious second phrase")


# ---------------------------------------------------------------------------
# Scenario 3 — note_off via note_on with velocity = 0 (MIDI running-status)
# ---------------------------------------------------------------------------

class TestScenario3_VelocityZeroNoteOff(unittest.TestCase):
    """
    Standard MIDI optimisation: a note_on message with velocity = 0 is
    treated as note_off (running-status shorthand).  MidiListener._callback
    must route these to on_note_off, not on_note_on.
    """

    def test_midi_listener_routes_vel0_as_note_off(self):
        """
        Injecting a raw 0x90/vel=0 event into _callback must call on_note_off
        and must NOT call on_note_on.
        """
        note_ons  = []
        note_offs = []
        listener = MidiListener(
            on_note_on=lambda p, v, t: note_ons.append((p, v)),
            on_note_off=lambda p, t: note_offs.append(p),
        )
        # 0x90 ch0 pitch=60 vel=0  →  note_off
        listener._callback(([0x90, 60, 0], 0.0), None)
        self.assertEqual(note_offs, [60],
                         "vel=0 note_on should be routed as note_off")
        self.assertEqual(note_ons, [],
                         "vel=0 note_on must NOT be routed as note_on")

    def test_midi_listener_routes_normal_note_on(self):
        """Sanity check: a normal note_on (vel > 0) still reaches on_note_on."""
        note_ons  = []
        note_offs = []
        listener = MidiListener(
            on_note_on=lambda p, v, t: note_ons.append((p, v)),
            on_note_off=lambda p, t: note_offs.append(p),
        )
        listener._callback(([0x90, 60, 80], 0.0), None)
        self.assertEqual(len(note_ons), 1)
        self.assertEqual(note_ons[0][0], 60)
        self.assertEqual(note_offs, [])

    def test_vel0_note_offs_complete_phrase_end_to_end(self):
        """
        End-to-end: note_offs arrive as 0x90/vel=0 events through MidiListener
        into PhraseDetector.  A complete phrase should fire normally.
        """
        det, phrases = make_detector()
        listener = MidiListener(
            on_note_on=lambda p, v, t: det.note_on(p, v, t),
            on_note_off=lambda p, t: det.note_off(p, t),
        )

        # note_on pitch=60, then vel=0 note_off, then same for pitch=62
        listener._callback(([0x90, 60, 80], 0.0), None)   # note_on  60
        time.sleep(0.10)                                    # hold 100 ms
        listener._callback(([0x90, 60, 0],  0.0), None)   # note_off 60 via vel=0
        listener._callback(([0x90, 62, 80], 0.0), None)   # note_on  62
        time.sleep(0.10)
        listener._callback(([0x90, 62, 0],  0.0), None)   # note_off 62 via vel=0

        fired = wait_for(lambda: len(phrases) == 1, timeout=0.5)
        self.assertTrue(fired,
                        "Phrase should complete when note_offs use vel=0 encoding")
        pitches = [n["pitch"] for n in phrases[0]]
        self.assertIn(60, pitches)
        self.assertIn(62, pitches)


if __name__ == "__main__":
    unittest.main(verbosity=2)

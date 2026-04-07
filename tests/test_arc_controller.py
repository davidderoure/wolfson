"""
Unit tests for ArcController proactive-play logic.

These tests exercise the timing guards in should_play_proactively() and the
touch_bass() / on_sax_played() timestamp methods without any MIDI hardware.

Strategy: manipulate _last_bass_time and _last_sax_time relative to the
real wall clock so that time_since_bass / time_since_sax inside
should_play_proactively() sees exactly the scenario under test.

Run with:
    cd ~/wolfson
    python -m pytest tests/test_arc_controller.py -v
or:
    python tests/test_arc_controller.py
"""

import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from controller.arc_controller import ArcController, PROACTIVE_MIN_INTERVAL, PROACTIVE_SILENCE_TRIGGER
from memory.phrase_memory import PhraseMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_controller(stage_offset_sec: float = 0.0) -> ArcController:
    """
    Return a started ArcController whose internal clock is wound to
    `stage_offset_sec` seconds into the arc.

    This is done by back-dating _start_time so that elapsed() returns
    exactly stage_offset_sec without actually waiting.
    """
    memory = PhraseMemory()
    arc = ArcController(memory)
    arc.start()
    # Backdate start so elapsed() == stage_offset_sec
    arc._start_time -= stage_offset_sec
    return arc


def set_bass_last_active(arc: ArcController, seconds_ago: float):
    """Set _last_bass_time as if bass last played `seconds_ago` seconds ago."""
    arc._last_bass_time = time.time() - seconds_ago


def set_sax_last_played(arc: ArcController, seconds_ago: float):
    """Set _last_sax_time as if sax last played `seconds_ago` seconds ago."""
    arc._last_sax_time = time.time() - seconds_ago


# ---------------------------------------------------------------------------
# Stage timing — from config so tests stay in sync with production values
# ---------------------------------------------------------------------------

from config import ARC

PEAK_START   = ARC["peak"][0]
PEAK_MID     = (ARC["peak"][0]   + ARC["peak"][1])   / 2
RES_START    = ARC["resolution"][0]
RES_MID      = (ARC["resolution"][0] + ARC["resolution"][1]) / 2
BUILD_MID    = (ARC["building"][0]   + ARC["building"][1])   / 2


# ---------------------------------------------------------------------------
# touch_bass tests
# ---------------------------------------------------------------------------

class TestTouchBass(unittest.TestCase):

    def test_touch_bass_updates_last_bass_time(self):
        """touch_bass() should refresh _last_bass_time to now."""
        arc = make_controller()
        # Wind back so bass looks old
        arc._last_bass_time = time.time() - 60.0
        before = arc._last_bass_time
        arc.touch_bass()
        self.assertGreater(arc._last_bass_time, before)
        self.assertAlmostEqual(arc._last_bass_time, time.time(), delta=0.1)

    def test_touch_bass_makes_time_since_bass_small(self):
        """After touch_bass(), time_since_bass should be near zero."""
        arc = make_controller()
        arc.touch_bass()
        time_since = time.time() - arc._last_bass_time
        self.assertLess(time_since, 0.1)


# ---------------------------------------------------------------------------
# PROACTIVE_MIN_INTERVAL gate
# ---------------------------------------------------------------------------

class TestMinInterval(unittest.TestCase):

    def test_blocks_when_sax_played_recently(self):
        """should_play_proactively() returns False within PROACTIVE_MIN_INTERVAL."""
        arc = make_controller(stage_offset_sec=RES_MID)
        arc._leadership = "sax"
        # Sax just played 0.5 s ago (well within 2.0 s minimum)
        set_sax_last_played(arc, 0.5)
        set_bass_last_active(arc, 999.0)   # bass silent for a long time
        self.assertFalse(arc.should_play_proactively())

    def test_allows_after_min_interval(self):
        """should_play_proactively() is not blocked once min interval has passed."""
        arc = make_controller(stage_offset_sec=RES_MID)
        arc._leadership = "sax"
        # Sax played PROACTIVE_MIN_INTERVAL + 1 s ago
        set_sax_last_played(arc, PROACTIVE_MIN_INTERVAL + 1.0)
        # Bass silent long enough for resolution trigger
        set_bass_last_active(arc, PROACTIVE_SILENCE_TRIGGER + 1.0)
        # In resolution stage this should now return True
        self.assertTrue(arc.should_play_proactively())


# ---------------------------------------------------------------------------
# Peak stage — bass-activity guard
# ---------------------------------------------------------------------------

class TestPeakStage(unittest.TestCase):

    def _peak_arc_with_sax_leadership(self) -> ArcController:
        arc = make_controller(stage_offset_sec=PEAK_MID)
        arc._leadership = "sax"
        return arc

    def test_peak_does_not_fire_while_bass_active(self):
        """
        In peak stage with sax leadership, should NOT fire if bass played
        within PROACTIVE_MIN_INTERVAL seconds.
        """
        arc = self._peak_arc_with_sax_leadership()
        set_sax_last_played(arc, PROACTIVE_SILENCE_TRIGGER * 0.7 + 0.5)
        # Bass active just 0.5 s ago — well within the 2 s guard
        set_bass_last_active(arc, 0.5)
        self.assertFalse(arc.should_play_proactively(),
                         "Sax should not fire proactively while bass is active in peak stage")

    def test_peak_fires_after_bass_pause(self):
        """
        In peak stage with sax leadership, SHOULD fire when bass has been
        silent for > PROACTIVE_MIN_INTERVAL and sax is past the trigger.
        """
        arc = self._peak_arc_with_sax_leadership()
        set_sax_last_played(arc, PROACTIVE_SILENCE_TRIGGER * 0.7 + 0.5)
        set_bass_last_active(arc, PROACTIVE_MIN_INTERVAL + 0.5)
        self.assertTrue(arc.should_play_proactively(),
                        "Sax should fire proactively after genuine bass pause in peak stage")

    def test_peak_bass_leadership_does_not_fire(self):
        """
        In peak stage with BASS leadership, should NOT fire (regardless of timing).
        """
        arc = make_controller(stage_offset_sec=PEAK_MID)
        arc._leadership = "bass"
        set_sax_last_played(arc, 999.0)
        set_bass_last_active(arc, 999.0)
        # The peak rule is gated on leadership == "sax"
        # The catch-all needs time_since_bass > 7.5 s — give it only 5 s
        set_bass_last_active(arc, 5.0)
        self.assertFalse(arc.should_play_proactively())


# ---------------------------------------------------------------------------
# Resolution stage — bass-activity guard
# ---------------------------------------------------------------------------

class TestResolutionStage(unittest.TestCase):

    def _resolution_arc(self) -> ArcController:
        return make_controller(stage_offset_sec=RES_MID)

    def test_resolution_does_not_fire_while_bass_active(self):
        """
        In resolution stage, should NOT fire if bass played recently.
        """
        arc = self._resolution_arc()
        set_sax_last_played(arc, PROACTIVE_SILENCE_TRIGGER + 0.5)
        # Bass active just 1 s ago — less than PROACTIVE_SILENCE_TRIGGER
        set_bass_last_active(arc, 1.0)
        self.assertFalse(arc.should_play_proactively(),
                         "Sax should not fire in resolution while bass is active")

    def test_resolution_fires_after_bass_silence(self):
        """
        In resolution stage, SHOULD fire when both sax and bass have been
        silent long enough.
        """
        arc = self._resolution_arc()
        set_sax_last_played(arc, PROACTIVE_SILENCE_TRIGGER + 0.5)
        set_bass_last_active(arc, PROACTIVE_SILENCE_TRIGGER + 0.5)
        self.assertTrue(arc.should_play_proactively(),
                        "Sax should fire in resolution after bass silence")


# ---------------------------------------------------------------------------
# Building stage
# ---------------------------------------------------------------------------

class TestBuildingStage(unittest.TestCase):

    def test_building_fires_after_long_bass_silence(self):
        """Building stage: fires when bass silent > PROACTIVE_SILENCE_TRIGGER * 1.5."""
        arc = make_controller(stage_offset_sec=BUILD_MID)
        set_sax_last_played(arc, PROACTIVE_MIN_INTERVAL + 0.5)
        set_bass_last_active(arc, PROACTIVE_SILENCE_TRIGGER * 1.5 + 0.5)
        self.assertTrue(arc.should_play_proactively())

    def test_building_does_not_fire_with_recent_bass(self):
        """Building stage: does NOT fire when bass still active."""
        arc = make_controller(stage_offset_sec=BUILD_MID)
        set_sax_last_played(arc, PROACTIVE_MIN_INTERVAL + 0.5)
        set_bass_last_active(arc, 1.0)   # bass played 1 s ago
        self.assertFalse(arc.should_play_proactively())


# ---------------------------------------------------------------------------
# Arc not started
# ---------------------------------------------------------------------------

class TestArcNotStarted(unittest.TestCase):

    def test_returns_false_before_start(self):
        """should_play_proactively() must return False if arc not yet started."""
        memory = PhraseMemory()
        arc = ArcController(memory)
        # Do NOT call arc.start()
        self.assertFalse(arc.should_play_proactively())


if __name__ == "__main__":
    unittest.main(verbosity=2)

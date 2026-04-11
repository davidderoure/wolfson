"""
HarmonyController — macro-level harmonic motion for the Wolfson improvisation system.

Four harmonic modes, selected by the arc controller based on performance stage:

  free        No harmonic steering. Chord index = NC_INDEX, chromatic scale bias.
              Used in the sparse opening before any harmonic flavour is established.

  modal       Choose a root + mode (e.g. D Dorian) and stay there.
              After MODAL_SHIFT_PHRASES phrases, shift the root by MODAL_SHIFT_SEMITONES.
              Good for long, meditative passages and the recapitulation stage.

  progression Step through a short chord progression (ii-V-I, VI-II-V-I, 12-bar blues).
              Each bass phrase advances one step. Tritone substitutions on V chords are
              applied with probability TRITONE_SUB_PROB. Used during building and peak.

  pedal       A fixed bass note (pedal tone) stays constant while the upper harmony
              cycles through related chords. Used for tension building and resolution.

Public API
----------
  controller = HarmonyController()
  controller.set_mode("modal", root=2, mode_name="dorian")   # optional configuration
  chord_idx, pitch_classes = controller.next_chord()          # call once per phrase
  controller.reset()

The returned `pitch_classes` is a frozenset[int] (0–11) suitable for the pitch-token
bias mask in phrase_generator._apply_scale_bias().
"""

import random

from data.chords import (
    NC_INDEX, CHORD_VOCAB_SIZE,
    QUAL_MAJOR, QUAL_DOM, QUAL_MINOR, QUAL_DIM,
    N_QUALITIES,
)
from data.scales import scale_pitch_classes, chord_to_mode, chord_root, tritone_sub as _tritone_sub, MODES


# ---------------------------------------------------------------------------
# Tunable constants
# ---------------------------------------------------------------------------

TRITONE_SUB_PROB    = 0.35   # probability of substituting V7 with its tritone sub
MODAL_SHIFT_PHRASES = 8      # advance root by MODAL_SHIFT_SEMITONES after this many phrases
MODAL_SHIFT_SEMITONES = 1    # semitone shift (up) for modal drift; use 5 for 4th, 7 for 5th
PEDAL_CYCLE_LENGTH  = 4      # how many phrases before pedal harmony moves to next chord


# ---------------------------------------------------------------------------
# Built-in chord progressions
# Each progression is a list of (root_offset_from_key, quality) pairs.
# root_offset is the semitone distance from the tonic root (0).
# ---------------------------------------------------------------------------

# ii-V-I  (minor 7 → dominant 7 → major 7)
PROG_II_V_I = [
    (2,  QUAL_MINOR),   # ii
    (7,  QUAL_DOM),     # V7
    (0,  QUAL_MAJOR),   # Imaj7
]

# I-VI-II-V  (rhythm changes turnaround; VI and II are dominant in bebop convention)
PROG_I_VI_II_V = [
    (0,  QUAL_MAJOR),   # I
    (9,  QUAL_DOM),     # VI7
    (2,  QUAL_DOM),     # II7
    (7,  QUAL_DOM),     # V7
]

# VI-II-V-I  (descending cycle-of-fifths approach)
PROG_VI_II_V_I = [
    (9,  QUAL_DOM),     # VI7
    (2,  QUAL_DOM),     # II7
    (7,  QUAL_DOM),     # V7
    (0,  QUAL_MAJOR),   # I
]

# Simplified 12-bar blues (4+4+4 = 12 bars, one chord per "bar" here)
PROG_BLUES_12 = [
    (0,  QUAL_DOM),   # I7   bar 1
    (0,  QUAL_DOM),   # I7   bar 2
    (0,  QUAL_DOM),   # I7   bar 3
    (0,  QUAL_DOM),   # I7   bar 4
    (5,  QUAL_DOM),   # IV7  bar 5
    (5,  QUAL_DOM),   # IV7  bar 6
    (0,  QUAL_DOM),   # I7   bar 7
    (0,  QUAL_DOM),   # I7   bar 8
    (7,  QUAL_DOM),   # V7   bar 9
    (5,  QUAL_DOM),   # IV7  bar 10
    (0,  QUAL_DOM),   # I7   bar 11
    (7,  QUAL_DOM),   # V7   bar 12 (turnaround)
]

NAMED_PROGRESSIONS = {
    "ii_v_i":    PROG_II_V_I,
    "i_vi_ii_v": PROG_I_VI_II_V,
    "vi_ii_v_i": PROG_VI_II_V_I,
    "blues":     PROG_BLUES_12,
}


# ---------------------------------------------------------------------------
# Helper: build a chord index from root + quality
# ---------------------------------------------------------------------------

def _chord_idx(root: int, quality: int) -> int:
    return (root % 12) * N_QUALITIES + quality


# ---------------------------------------------------------------------------
# HarmonyController
# ---------------------------------------------------------------------------

class HarmonyController:
    """
    Tracks harmonic state across the performance and issues chord tokens
    for the phrase generator.

    Call next_chord() once each time the sax is about to respond.
    It returns (chord_idx, pitch_classes) where pitch_classes is a
    frozenset[int] of allowed pitch classes (0–11) for the scale bias.
    """

    def __init__(self):
        self._mode          = "free"    # free | modal | progression | pedal
        self._key_root      = 0         # tonic pitch class (0=C … 11=B)

        # Modal state
        self._modal_root    = 2         # current modal root (default D)
        self._modal_name    = "dorian"  # current mode name
        self._modal_phrases = 0         # phrases played in current modal position

        # Progression state
        self._prog_name     = "ii_v_i"
        self._prog_step     = 0

        # Pedal state
        self._pedal_root    = 0         # pedal tone (bass note stays here)
        self._pedal_harmony = None      # list of chord indices cycling above the pedal
        self._pedal_step    = 0

    # -----------------------------------------------------------------------
    # Configuration
    # -----------------------------------------------------------------------

    def set_mode(
        self,
        mode:           str,
        key_root:       int  = None,
        prog_name:      str  = None,
        modal_root:     int  = None,
        modal_name:     str  = None,
        pedal_root:     int  = None,
    ):
        """
        Switch harmonic mode and configure its parameters.

        mode:       'free' | 'modal' | 'progression' | 'pedal'
        key_root:   tonic pitch class for progressions and pedal
        prog_name:  'ii_v_i' | 'i_vi_ii_v' | 'vi_ii_v_i' | 'blues'
        modal_root: root pitch class for modal playing
        modal_name: mode name from data.scales.MODES
        pedal_root: pitch class for the pedal tone
        """
        self._mode = mode
        if key_root   is not None:  self._key_root    = key_root % 12
        if prog_name  is not None:  self._prog_name   = prog_name
        if modal_root is not None:  self._modal_root  = modal_root % 12
        if modal_name is not None:  self._modal_name  = modal_name
        if pedal_root is not None:  self._pedal_root  = pedal_root % 12
        # Re-initialise pedal harmony when entering pedal mode
        if mode == "pedal":
            self._pedal_harmony = self._build_pedal_harmony()
            self._pedal_step    = 0

    def reset(self):
        """Return to free mode, clearing all state."""
        self.__init__()

    # -----------------------------------------------------------------------
    # Main API
    # -----------------------------------------------------------------------

    def next_chord(self, arc_position: float = 0.0) -> tuple[int, frozenset]:
        """
        Advance harmonic state by one phrase and return (chord_idx, pitch_classes).

        pitch_classes is a frozenset[int] (0–11) for scale-tone bias.
        arc_position: 0.0–1.0 fraction of the total arc elapsed; used by pedal
                      mode to ensure the final phrase always resolves to i minor.
        """
        if self._mode == "free":
            return self._next_free()
        elif self._mode == "modal":
            return self._next_modal()
        elif self._mode == "progression":
            return self._next_progression()
        elif self._mode == "pedal":
            return self._next_pedal(arc_position)
        else:
            return self._next_free()

    def current_mode_name(self) -> str:
        return self._mode

    # -----------------------------------------------------------------------
    # Free mode
    # -----------------------------------------------------------------------

    def _next_free(self) -> tuple[int, frozenset]:
        return NC_INDEX, frozenset(range(12))

    # -----------------------------------------------------------------------
    # Modal mode
    # -----------------------------------------------------------------------

    def _next_modal(self) -> tuple[int, frozenset]:
        """
        Stay on the current modal root / mode.
        After MODAL_SHIFT_PHRASES phrases, drift the root up by MODAL_SHIFT_SEMITONES.
        """
        self._modal_phrases += 1
        if self._modal_phrases >= MODAL_SHIFT_PHRASES:
            self._modal_root    = (self._modal_root + MODAL_SHIFT_SEMITONES) % 12
            self._modal_phrases = 0

        # Choose chord quality that best represents the mode
        quality   = _mode_to_quality(self._modal_name)
        cid       = _chord_idx(self._modal_root, quality)
        pc        = scale_pitch_classes(self._modal_root, self._modal_name)
        return cid, pc

    # -----------------------------------------------------------------------
    # Progression mode
    # -----------------------------------------------------------------------

    def _next_progression(self) -> tuple[int, frozenset]:
        """
        Step through the chosen progression; advance one step per phrase.
        Apply tritone substitution on dominant chords with TRITONE_SUB_PROB.
        """
        prog = NAMED_PROGRESSIONS.get(self._prog_name, PROG_II_V_I)
        offset, quality = prog[self._prog_step % len(prog)]
        self._prog_step = (self._prog_step + 1) % len(prog)

        root = (self._key_root + offset) % 12
        cid  = _chord_idx(root, quality)

        # Tritone substitution on dominant chords
        if quality == QUAL_DOM and random.random() < TRITONE_SUB_PROB:
            cid = _tritone_sub(cid)

        mode_name = chord_to_mode(cid)
        pc        = scale_pitch_classes(chord_root(cid), mode_name)
        return cid, pc

    # -----------------------------------------------------------------------
    # Pedal mode
    # -----------------------------------------------------------------------

    def _build_pedal_harmony(self) -> list[int]:
        """
        Construct a list of chord indices that cycle above the pedal tone.

        The pedal root stays constant in the bass while the upper harmony moves
        through related chords — a common jazz device (e.g. D pedal while
        harmony moves through Dm7, G7/D, Dm7, A7/D).

        We use:  tonic minor → bVII dominant → tonic minor → V dominant
        (a four-chord loop, all rooted at or resolving to the pedal root)
        """
        r = self._pedal_root
        return [
            _chord_idx(r,           QUAL_MINOR),   # i minor (pedal root)
            _chord_idx((r + 10) % 12, QUAL_DOM),   # bVII7
            _chord_idx(r,           QUAL_MINOR),   # i minor
            _chord_idx((r + 7) % 12,  QUAL_DOM),   # V7 (tension before return)
        ]

    def _next_pedal(self, arc_position: float = 0.0) -> tuple[int, frozenset]:
        """
        The bass note stays on the pedal root; the chord cycles through
        pedal harmony every PEDAL_CYCLE_LENGTH phrases.

        When arc_position > 0.9 the tonic chord (i minor) is forced so the
        performance always resolves to home regardless of tempo or phrase count.
        """
        if not self._pedal_harmony:
            self._pedal_harmony = self._build_pedal_harmony()

        phrase_in_cycle = self._pedal_step % PEDAL_CYCLE_LENGTH
        chord_in_list   = self._pedal_step // PEDAL_CYCLE_LENGTH % len(self._pedal_harmony)

        if phrase_in_cycle == 0:
            # Advance to next chord in the pedal loop
            pass   # chord_in_list already computed correctly

        self._pedal_step += 1

        # Near the end of the arc always resolve to i minor regardless of
        # where the cycle happens to be — ensures a tonic landing at any tempo.
        if arc_position > 0.9:
            chord_in_list = 0   # index 0 = i minor (tonic)

        cid       = self._pedal_harmony[chord_in_list]
        mode_name = chord_to_mode(cid)
        pc        = scale_pitch_classes(chord_root(cid), mode_name)
        return cid, pc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mode_to_quality(mode_name: str) -> int:
    """Return the most representative chord quality for a given mode."""
    _map = {
        "ionian":      QUAL_MAJOR,
        "lydian":      QUAL_MAJOR,
        "bebop_major": QUAL_MAJOR,
        "dorian":      QUAL_MINOR,
        "phrygian":    QUAL_MINOR,
        "aeolian":     QUAL_MINOR,
        "locrian":     QUAL_DIM,
        "mixolydian":  QUAL_DOM,
        "lydian_dom":  QUAL_DOM,
        "altered":     QUAL_DOM,
        "bebop_dom":   QUAL_DOM,
        "blues":       QUAL_DOM,
        "whole_tone":  QUAL_DOM,
        "diminished":  QUAL_DIM,
    }
    return _map.get(mode_name, QUAL_MINOR)


# ---------------------------------------------------------------------------
# Stage → harmonic mode mapping (used by arc_controller)
# ---------------------------------------------------------------------------

def stage_to_harmonic_mode(stage: str) -> str:
    """
    Suggest a harmonic mode for each performance stage.

    sparse         → free   (no harmonic constraints at the start)
    building       → modal  (settle into a mode)
    peak           → progression (active chord motion)
    recapitulation → modal  (return to the original modal feel)
    resolution     → pedal  (pedal tone → sense of settling)
    """
    return {
        "sparse":         "free",
        "building":       "modal",
        "peak":           "progression",
        "recapitulation": "modal",
        "resolution":     "pedal",
    }.get(stage, "free")

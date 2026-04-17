"""
Jazz scale and mode definitions.

Scales are stored as ordered interval lists (semitones above root).
scale_pitch_classes(root, mode) returns a frozenset of pitch classes (0-11)
suitable for use as a pitch-token bias mask in the generator.
"""

from data.chords import QUAL_MAJOR, QUAL_DOM, QUAL_MINOR, QUAL_DIM

# ------------------------------------------------------------------
# Mode interval tables
# ------------------------------------------------------------------

MODES: dict[str, list[int]] = {
    # Diatonic
    "ionian":        [0, 2, 4, 5, 7, 9, 11],   # major
    "dorian":        [0, 2, 3, 5, 7, 9, 10],   # minor with ♮6 — classic jazz minor
    "phrygian":      [0, 1, 3, 5, 7, 8, 10],
    "lydian":        [0, 2, 4, 6, 7, 9, 11],   # major with ♯4
    "mixolydian":    [0, 2, 4, 5, 7, 9, 10],   # dominant — major with ♭7
    "aeolian":       [0, 2, 3, 5, 7, 8, 10],   # natural minor
    "locrian":       [0, 1, 3, 5, 6, 8, 10],

    # Jazz-specific
    "lydian_dom":    [0, 2, 4, 6, 7, 9, 10],   # Lydian ♭7 — tritone sub sound
    "altered":       [0, 1, 3, 4, 6, 8, 10],   # 7th mode of melodic minor
    "bebop_dom":     [0, 2, 4, 5, 7, 9, 10, 11],  # mixolydian + ♮7 passing tone
    "bebop_major":   [0, 2, 4, 5, 7, 8, 9, 11],   # major + ♭6 passing tone
    "blues":         [0, 3, 5, 6, 7, 10],       # minor blues hexatonic
    "whole_tone":    [0, 2, 4, 6, 8, 10],       # symmetric; suits aug/dominant
    "diminished":    [0, 2, 3, 5, 6, 8, 9, 11], # half-whole diminished
    "chromatic":     list(range(12)),            # unconstrained
}

# Default mode to use when improvising over each chord quality
_QUALITY_MODES: dict[int, str] = {
    QUAL_MAJOR: "ionian",
    QUAL_DOM:   "mixolydian",
    QUAL_MINOR: "dorian",
    QUAL_DIM:   "diminished",
}


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

def scale_pitch_classes(root: int, mode: str) -> frozenset[int]:
    """
    Return the frozenset of pitch classes (0-11) for a given root and mode.
    root: 0=C, 1=C♯/D♭, ..., 11=B
    """
    intervals = MODES.get(mode, MODES["chromatic"])
    return frozenset((root + i) % 12 for i in intervals)


def chord_to_mode(chord_idx: int) -> str:
    """
    Return a reasonable default mode for improvising over a chord.
    chord_idx: combined index from data/chords.py (root * N_QUALITIES + quality)
    """
    from data.chords import NC_INDEX, N_QUALITIES
    if chord_idx == NC_INDEX:
        return "chromatic"
    quality = chord_idx % N_QUALITIES
    return _QUALITY_MODES.get(quality, "chromatic")


def chord_root(chord_idx: int) -> int:
    """Extract the root pitch class (0-11) from a chord index."""
    from data.chords import NC_INDEX, N_QUALITIES
    if chord_idx == NC_INDEX:
        return 0
    return chord_idx // N_QUALITIES


def chord_tones(chord_idx: int) -> frozenset:
    """
    Return the key chord tones (root, 3rd, 7th) as pitch classes (0–11).

    These are the harmonically important tones for voice-leading endpoint bias:
    the generator is nudged toward landing on them at phrase cadences.

      Major   → root, maj3 (+4), maj7 (+11)
      Dominant→ root, maj3 (+4), min7 (+10)
      Minor   → root, min3 (+3), min7 (+10)
      Diminished→ root, min3 (+3), dim7 (+9)

    Returns frozenset() for NC_INDEX (no harmonic constraint).
    """
    from data.chords import NC_INDEX, N_QUALITIES
    if chord_idx == NC_INDEX:
        return frozenset()
    root    = chord_idx // N_QUALITIES
    quality = chord_idx % N_QUALITIES
    intervals = {
        QUAL_MAJOR: (0, 4, 11),
        QUAL_DOM:   (0, 4, 10),
        QUAL_MINOR: (0, 3, 10),
        QUAL_DIM:   (0, 3,  9),
    }.get(quality, (0, 4, 7))
    return frozenset((root + i) % 12 for i in intervals)


def identify_mode(root: int, pcs: frozenset[int]) -> tuple[str, float]:
    """
    Return (mode_name, confidence) for the best-matching mode given a root
    and an observed frozenset of pitch classes.

    Confidence is a precision score: the fraction of observed pcs that are
    contained within the candidate mode.  Among equal-precision candidates,
    the most specific mode (fewest scale degrees) is preferred — so "blues"
    wins over "dorian" when both cover the same observed notes perfectly.

    Bebop scales and chromatic are excluded: their large note counts make
    them overfit every partial observation.  Returns ("chromatic", 0.0) when
    pcs is empty or no candidate scores above zero.
    """
    if not pcs:
        return ("chromatic", 0.0)

    # Ordered by jazz prevalence so ties resolve toward the idiomatic choice.
    CANDIDATES = [
        "dorian", "mixolydian", "ionian", "aeolian", "lydian",
        "lydian_dom", "altered", "phrygian", "locrian",
        "blues", "whole_tone", "diminished",
    ]

    best_mode  = "chromatic"
    best_score = -1.0
    best_size  = 13

    for name in CANDIDATES:
        expected  = scale_pitch_classes(root, name)
        precision = len(pcs & expected) / len(pcs)
        size      = len(expected)
        if precision > best_score or (precision == best_score and size < best_size):
            best_score = precision
            best_mode  = name
            best_size  = size

    return (best_mode, best_score)


def tritone_sub(chord_idx: int) -> int:
    """
    Return the tritone substitution of a chord index.
    Only meaningful for dominant chords (quality 1); for others returns the
    original index unchanged.
    The tritone sub of X7 is (X+6)7 — same tritone interval, different root.
    """
    from data.chords import NC_INDEX, N_QUALITIES, QUAL_DOM
    if chord_idx == NC_INDEX:
        return chord_idx
    root    = chord_idx // N_QUALITIES
    quality = chord_idx % N_QUALITIES
    if quality != QUAL_DOM:
        return chord_idx
    sub_root = (root + 6) % 12
    return sub_root * N_QUALITIES + QUAL_DOM

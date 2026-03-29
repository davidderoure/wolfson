"""
Real-time analysis of bass phrases.

Extracts features used by the arc controller to decide:
  - whether the bassist is leading or filling
  - what rhetorical type the phrase is (question vs answer)
  - what contour the sax response should aim for
  - what rhythmic feel the phrase has (straight vs swing/triplet)

Swing detection
---------------
Jazz swing is a 12/8 subdivision: the beat divides into three equal parts
(triplet 8ths), and typical "swung 8th notes" are the 1st and 3rd of each
triplet — giving a 2:1 long-short ratio within each beat.

We detect this by examining consecutive IOI pairs (adjacent gaps between
note onsets). In a swung phrase, pairs of IOIs that sum to approximately
one beat will have a ratio close to 2:1. In a straight feel, the ratio is
close to 1:1.

`swing_ratio` in the returned feature dict:
  ~1.0   straight feel (8th notes, quarter notes, no swing)
  ~2.0   triplet swing (classic jazz swing 8ths)
  ~0.5   reverse swing (unusual, short-long pattern)

`rhythmic_feel` is a string label: 'swing', 'straight', or 'mixed'.
"""

import math


# Thresholds (tune by ear during rehearsal)
SPARSE_DENSITY_THRESHOLD  = 1.5   # notes/sec below this → bassist is comping/filling
MELODIC_AMBITUS_THRESHOLD = 7     # semitones; below this → phrase is not strongly directional
QUESTION_END_THRESHOLD    = 0.15  # end pitch above this fraction of ambitus above mean → question

# Swing detection
BEAT_PAIR_TOLERANCE   = 0.15   # two IOIs are "a beat pair" if their sum is within
                               # this fraction of the estimated beat duration
SWING_RATIO_THRESHOLD = 1.5    # ratio > this → swing feel; < 1/this → reverse swing
MIN_PAIRS_FOR_SWING   = 2      # need at least this many pairs to classify


def analyze(phrase: list[dict]) -> dict:
    """
    Analyze a phrase and return a feature dict.

    phrase: list of note dicts with keys: pitch (int), onset (float), offset (float)

    Returns dict with:
        note_density     float    notes per second
        ambitus          int      pitch range (semitones)
        mean_pitch       float
        contour_slope    float    linear regression slope over pitch sequence (positive = rising)
        end_pitch        int      pitch of last note
        end_direction    int      +1 rising, -1 falling, 0 same (last two notes)
        rhetorical_type  str      'question', 'answer', or 'neutral'
        is_sparse        bool     bassist is comping / leaving space
        duration_sec     float    total phrase duration
    """
    if not phrase:
        return _neutral()

    pitches = [n["pitch"] for n in phrase]
    onsets  = [n["onset"] for n in phrase]
    offsets = [n["offset"] for n in phrase]

    duration_sec = offsets[-1] - onsets[0]
    note_density = len(phrase) / max(duration_sec, 0.01)

    ambitus    = max(pitches) - min(pitches)
    mean_pitch = sum(pitches) / len(pitches)
    end_pitch  = pitches[-1]

    contour_slope = _linear_slope(pitches)

    if len(pitches) >= 2:
        diff = pitches[-1] - pitches[-2]
        end_direction = 1 if diff > 0 else (-1 if diff < 0 else 0)
    else:
        end_direction = 0

    rhetorical_type = _classify_rhetorical(
        pitches, mean_pitch, ambitus, end_pitch, end_direction, contour_slope
    )

    swing_ratio, rhythmic_feel = _detect_swing(onsets)

    velocities    = [n["velocity"] for n in phrase if "velocity" in n]
    mean_velocity = int(sum(velocities) / len(velocities)) if velocities else 64

    # Pitch-class set: which of the 12 chromatic pitch classes appeared.
    # Used by the arc controller to steer the sax toward the bassist's
    # harmonic language when the phrase is tonally unambiguous.
    bass_pitch_classes = frozenset(n["pitch"] % 12 for n in phrase)

    return {
        "note_density":      note_density,
        "ambitus":           ambitus,
        "mean_pitch":        mean_pitch,
        "contour_slope":     contour_slope,
        "end_pitch":         end_pitch,
        "end_direction":     end_direction,
        "rhetorical_type":   rhetorical_type,
        "is_sparse":         note_density < SPARSE_DENSITY_THRESHOLD,
        "duration_sec":      duration_sec,
        "swing_ratio":       swing_ratio,
        "rhythmic_feel":     rhythmic_feel,
        "mean_velocity":     mean_velocity,
        "bass_pitch_classes": bass_pitch_classes,
    }


def complement_contour(features: dict) -> str:
    """
    Return the target contour for the sax response.

    Jazz dialogue principle: answer a question with an answer, and vice versa.
    Also: mirror a strong directional phrase with its opposite for balance.
    """
    rtype = features["rhetorical_type"]
    if rtype == "question":
        return "answer"   # sax should resolve
    if rtype == "answer":
        return "question" # sax should open the next exchange
    # Neutral: complement the contour slope
    return "descending" if features["contour_slope"] > 0 else "ascending"


def _classify_rhetorical(pitches, mean_pitch, ambitus, end_pitch, end_direction, slope):
    if ambitus < MELODIC_AMBITUS_THRESHOLD:
        return "neutral"

    # End pitch relative to phrase range
    end_relative = (end_pitch - mean_pitch) / max(ambitus, 1)

    # Question: ends high, or rising at the end
    # Answer:   ends low, or falling at the end
    question_score = (0.6 * end_relative) + (0.4 * end_direction)

    if question_score > QUESTION_END_THRESHOLD:
        return "question"
    elif question_score < -QUESTION_END_THRESHOLD:
        return "answer"
    return "neutral"


def _linear_slope(values: list) -> float:
    """Slope of a linear regression through the value sequence."""
    n = len(values)
    if n < 2:
        return 0.0
    xs = list(range(n))
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num   = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, values))
    denom = sum((x - x_mean) ** 2 for x in xs)
    return num / denom if denom else 0.0


def _detect_swing(onsets: list[float]) -> tuple[float, str]:
    """
    Estimate the swing ratio from a sequence of note onsets.

    Returns (swing_ratio, rhythmic_feel).

    Strategy: for each consecutive IOI pair (a, b), compute max(a,b)/min(a,b).
    In swing feel the IOIs alternate long-short (2:1), giving a consistent
    ratio of 2.0. In straight feel all IOIs are equal, giving 1.0. This does
    not require a beat estimate and is robust to different subdivisions.

    Note: triplet 8ths (all equal) are indistinguishable from straight 8ths
    by this metric alone — both give ratio 1.0. That is acceptable here since
    triplet 8ths do not need additional swing bias applied.
    """
    if len(onsets) < 4:
        return 1.0, "mixed"

    iois = [onsets[i + 1] - onsets[i] for i in range(len(onsets) - 1)]

    # Drop outlier gaps (phrase rests) using median as reference
    sorted_iois = sorted(iois)
    median_ioi  = sorted_iois[len(sorted_iois) // 2]
    iois = [x for x in iois if 0 < x < median_ioi * 3.5]

    if len(iois) < 3:
        return 1.0, "mixed"

    # max/min ratio for each consecutive IOI pair
    ratios = []
    for i in range(len(iois) - 1):
        a, b = iois[i], iois[i + 1]
        lo = min(a, b)
        if lo > 0:
            ratios.append(max(a, b) / lo)

    if len(ratios) < MIN_PAIRS_FOR_SWING:
        return 1.0, "mixed"

    swing_ratio = sum(ratios) / len(ratios)

    if swing_ratio > SWING_RATIO_THRESHOLD:
        feel = "swing"
    else:
        feel = "straight"

    return swing_ratio, feel


def _neutral() -> dict:
    return {
        "note_density":      0.0,
        "ambitus":           0,
        "mean_pitch":        60.0,
        "contour_slope":     0.0,
        "end_pitch":         60,
        "end_direction":     0,
        "rhetorical_type":   "neutral",
        "is_sparse":         True,
        "duration_sec":      0.0,
        "swing_ratio":       1.0,
        "rhythmic_feel":     "mixed",
        "mean_velocity":     64,
        "bass_pitch_classes": frozenset(),
    }

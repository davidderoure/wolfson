"""
Real-time analysis of bass phrases.

Extracts features used by the arc controller to decide:
  - whether the bassist is leading or filling
  - what rhetorical type the phrase is (question vs answer)
  - what contour the sax response should aim for
"""

import math


# Thresholds (tune by ear during rehearsal)
SPARSE_DENSITY_THRESHOLD  = 2.5   # notes/sec below this → bassist is comping/filling
MELODIC_AMBITUS_THRESHOLD = 7     # semitones; below this → phrase is not strongly directional
QUESTION_END_THRESHOLD    = 0.15  # end pitch above this fraction of ambitus above mean → question


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

    return {
        "note_density":    note_density,
        "ambitus":         ambitus,
        "mean_pitch":      mean_pitch,
        "contour_slope":   contour_slope,
        "end_pitch":       end_pitch,
        "end_direction":   end_direction,
        "rhetorical_type": rhetorical_type,
        "is_sparse":       note_density < SPARSE_DENSITY_THRESHOLD,
        "duration_sec":    duration_sec,
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


def _neutral() -> dict:
    return {
        "note_density":    0.0,
        "ambitus":         0,
        "mean_pitch":      60.0,
        "contour_slope":   0.0,
        "end_pitch":       60,
        "end_direction":   0,
        "rhetorical_type": "neutral",
        "is_sparse":       True,
        "duration_sec":    0.0,
    }

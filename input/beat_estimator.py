"""
Real-time tempo estimation from live bass note onsets.

Algorithm:
  1. Maintain a sliding window of recent note onsets.
  2. Compute consecutive inter-onset intervals (IOIs).
  3. Find the modal IOI — the most common inter-note gap.
  4. Convert to BPM: raw_bpm = 60 / modal_ioi.
  5. Halve or double into the sweet spot (TEMPO_SWEET_SPOT), resolving the
     octave ambiguity (8th vs quarter note interpretation) deterministically.
  6. Apply exponential smoothing for stability.

This approach avoids a proximity-to-prior bias, which creates attractors near
the initial estimate. Instead, any tempo in the sweet spot is accepted directly;
tempos outside it are octave-folded in. The sweet spot (55–220 BPM) covers
all practical jazz bass tempos when interpreted as quarter notes.

Config knob: TEMPO_HINT_BPM in config.py. If the detected tempo deviates by
more than an octave from this hint, the octave-corrected value closest to the
hint is preferred. Leave at 0 to disable the hint.
"""

import time

# BPM range outside which we octave-fold
TEMPO_SWEET_SPOT = (55, 220)

# Exponential smoothing factor (0 = raw/fast, 1 = frozen)
SMOOTHING = 0.25

# Histogram bin width (seconds)
BIN_SIZE = 0.020   # 20 ms

# Minimum onsets before attempting estimation
MIN_ONSETS = 8

# Sliding window size
WINDOW = 32


class BeatEstimator:
    def __init__(self, initial_bpm: float = 120.0, hint_bpm: float = 0.0):
        """
        initial_bpm: starting tempo guess
        hint_bpm:    if non-zero, bias octave resolution toward this tempo
                     (set to your approximate performance tempo in config.py)
        """
        self._onsets: list[float] = []
        self._bpm   = initial_bpm
        self._hint  = hint_bpm

    def note_on(self, t: float = None):
        t = t if t is not None else time.time()
        self._onsets.append(t)
        if len(self._onsets) > WINDOW:
            self._onsets.pop(0)
        if len(self._onsets) >= MIN_ONSETS:
            self._bpm = self._estimate()

    @property
    def bpm(self) -> float:
        return self._bpm

    @property
    def beat_duration(self) -> float:
        return 60.0 / self._bpm

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def _estimate(self) -> float:
        iois = [
            self._onsets[i + 1] - self._onsets[i]
            for i in range(len(self._onsets) - 1)
        ]
        # Ignore gaps longer than a whole note at MIN_BPM (phrase rests)
        iois = [x for x in iois if 0.04 <= x <= 60.0 / TEMPO_SWEET_SPOT[0]]
        if not iois:
            return self._bpm

        modal_ioi = _modal_ioi(iois)
        raw_bpm   = 60.0 / modal_ioi

        # Octave-fold into the sweet spot
        bpm = _fold_into_sweet_spot(raw_bpm, self._hint or self._bpm)

        return SMOOTHING * self._bpm + (1.0 - SMOOTHING) * bpm


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _modal_ioi(iois: list[float]) -> float:
    """Centre of the tallest IOI histogram bin."""
    max_ioi = 60.0 / TEMPO_SWEET_SPOT[0]
    n_bins  = int(max_ioi / BIN_SIZE) + 1
    hist    = [0] * n_bins

    for ioi in iois:
        b = int(ioi / BIN_SIZE)
        if 0 <= b < n_bins:
            hist[b] += 1

    peak_bin = max(range(n_bins), key=lambda i: hist[i])
    return (peak_bin + 0.5) * BIN_SIZE


def _fold_into_sweet_spot(bpm: float, reference: float) -> float:
    """
    Repeatedly halve or double `bpm` until it lands in TEMPO_SWEET_SPOT.
    When multiple octave positions are valid, choose the one closest to
    `reference` (the hint or current running estimate).
    """
    lo, hi = TEMPO_SWEET_SPOT

    # Collect all valid octave positions
    candidates = []
    v = bpm
    for _ in range(6):
        if lo <= v <= hi:
            candidates.append(v)
        if v < lo:
            v *= 2
        else:
            v /= 2

    if not candidates:
        # Clamp to nearest boundary
        return lo if bpm < lo else hi

    # Pick the candidate closest to the reference in log space
    import math
    return min(candidates, key=lambda c: abs(math.log(c / reference)))

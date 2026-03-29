"""
Macro-level structural arc controller.

Manages:
  - The 5-minute performance arc (sparse → building → peak → recap → resolution)
  - Leadership state: who is initiating (bass or sax)
  - Proactive mode: sax plays without waiting for a bass phrase to end
  - Response parameters: generation length, temperature, contour target, recall mode
"""

import random
import time

from config import ARC
from data.chords import NC_INDEX
from input.phrase_analyzer import analyze, complement_contour
from memory.phrase_memory import PhraseMemory
from controller.harmony import HarmonyController, stage_to_harmonic_mode


# How long the sax waits in proactive mode before initiating (seconds)
PROACTIVE_SILENCE_TRIGGER = 3.0

# Minimum gap between sax phrases in proactive mode (avoid crowding)
PROACTIVE_MIN_INTERVAL = 2.0


class ArcController:

    STAGES = ["sparse", "building", "peak", "recapitulation", "resolution"]

    def __init__(self, memory: PhraseMemory):
        self.memory = memory

        self._start_time       = None
        self._last_bass_time   = None   # wall time of last bass phrase completion
        self._last_sax_time    = None   # wall time of last sax phrase played
        self._last_bass_features = None

        # Leadership state: 'bass' or 'sax'
        # Updated by _update_leadership() each time a bass phrase arrives.
        self._leadership = "bass"

        # Harmonic controller — issues chord tokens and scale pitch-class sets
        self._harmony = HarmonyController()
        self._last_harmonic_stage = None   # detect stage changes

    # -----------------------------------------------------------------------
    # Lifecycle
    # -----------------------------------------------------------------------

    def start(self):
        self._start_time     = time.time()
        self._last_bass_time = time.time()
        self._last_sax_time  = time.time()

    def elapsed(self) -> float:
        return time.time() - self._start_time if self._start_time else 0.0

    def stage(self) -> str:
        t = self.elapsed()
        for name in self.STAGES:
            start, end = ARC[name]
            if start <= t < end:
                return name
        return "resolution"

    # -----------------------------------------------------------------------
    # Bass phrase arrival
    # -----------------------------------------------------------------------

    def on_bass_phrase(self, phrase: list[dict]) -> dict:
        """
        Called when the phrase detector fires. Returns a response_params dict
        for the generator. Also updates internal leadership state.
        """
        features = analyze(phrase)
        self._last_bass_time     = time.time()
        self._last_bass_features = features
        self._update_leadership(features)
        return self._build_params(phrase, features, proactive=False)

    def on_sax_played(self):
        """Called by main.py whenever the sax finishes playing a phrase."""
        self._last_sax_time = time.time()

    # -----------------------------------------------------------------------
    # Proactive mode
    # -----------------------------------------------------------------------

    def should_play_proactively(self) -> bool:
        """
        True when the sax should initiate without waiting for a bass phrase.

        Conditions:
          - We are in a stage where sax leadership is expected, OR
          - Bass has been silent long enough that it's clearly inviting the sax
          - A minimum interval since the sax last played has elapsed
        """
        if self._start_time is None:
            return False

        stage = self.stage()
        elapsed = self.elapsed()

        time_since_bass = (time.time() - self._last_bass_time) if self._last_bass_time else 999
        time_since_sax  = (time.time() - self._last_sax_time)  if self._last_sax_time  else 999

        if time_since_sax < PROACTIVE_MIN_INTERVAL:
            return False

        # Resolution: sax always gets the last word
        if stage == "resolution" and time_since_sax > PROACTIVE_SILENCE_TRIGGER:
            return True

        # Peak: sax occasionally interrupts / initiates
        if stage == "peak" and self._leadership == "sax":
            return time_since_sax > PROACTIVE_SILENCE_TRIGGER * 0.7

        # Building: sax initiates when bassist leaves a long gap
        if stage == "building" and time_since_bass > PROACTIVE_SILENCE_TRIGGER * 1.5:
            return True

        # Any stage: bassist silent for a long time → fill
        if time_since_bass > PROACTIVE_SILENCE_TRIGGER * 2.5:
            return True

        return False

    def get_proactive_params(self) -> dict:
        """
        Build generation params for a sax-initiated phrase.
        Seed from sax memory (continue its own line) or early bass (recapitulate).
        """
        stage = self.stage()

        if stage == "recapitulation":
            early = self.memory.recall_early("bass", n=4)
            seed  = random.choice(early) if early else self.memory.recall_random("sax")
        else:
            seed = self.memory.recall_recent("sax", n=1)
            seed = seed[0] if seed else self.memory.recall_random("bass")

        if seed is None:
            return None

        return self._build_params(seed, analyze(seed), proactive=True)

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _update_leadership(self, features: dict):
        """
        Decide who is leading based on bass phrase features and stage.

        Sparse bass (low density, small ambitus) → bassist is comping → sax leads.
        Dense, wide-range bass → bassist is soloing → sax follows.
        Stage provides the macro context for how much weight to give this.
        """
        stage = self.stage()

        # Stage-level defaults
        stage_wants_sax = stage in ("recapitulation", "resolution")
        stage_wants_bass = stage in ("sparse",)

        if stage_wants_bass:
            self._leadership = "bass"
            return
        if stage_wants_sax:
            self._leadership = "sax"
            return

        # For building and peak: use phrase features to decide
        is_sparse  = features["is_sparse"]
        small_range = features["ambitus"] < 7

        if is_sparse and small_range:
            self._leadership = "sax"   # bassist is comping, sax should lead
        elif not is_sparse and not small_range:
            self._leadership = "bass"  # bassist is soloing, sax should respond
        else:
            # Gradual drift toward sax leadership during peak
            if stage == "peak":
                self._leadership = "sax" if random.random() < 0.6 else "bass"
            else:
                self._leadership = "bass"

    def _build_params(self, phrase: list[dict], features: dict, proactive: bool) -> dict:
        stage = self.stage()

        # Switch harmonic mode when the performance stage changes
        if stage != self._last_harmonic_stage:
            new_harm_mode = stage_to_harmonic_mode(stage)
            self._harmony.set_mode(new_harm_mode)
            self._last_harmonic_stage = stage

        # Advance harmony and get chord token + scale pitch classes
        chord_idx, scale_pcs = self._harmony.next_chord()

        contour_target = complement_contour(features)

        # Phrase length: longer when sax leads, shorter when following
        sax_leads = (self._leadership == "sax") or proactive
        base_len  = _stage_base_length(stage)
        n_notes   = int(base_len * (1.3 if sax_leads else 0.8))
        n_notes   = max(3, n_notes)

        temperature = _stage_temperature(stage)

        # Recall logic: use memory more as the piece develops
        use_recall = _should_recall(stage, self.memory)
        if use_recall:
            if stage == "recapitulation":
                recalled = self.memory.recall_early("bass", n=4)
                seed = random.choice(recalled) if recalled else phrase
            else:
                seed = self.memory.recall_recent("bass", n=1)
                seed = seed[0] if seed else phrase
            mode = "recall"
        else:
            seed = phrase
            mode = "generate"

        # Swing bias: complement the bass rhythmic feel.
        # Straight/sparse bass → triplet response (contrast).
        # Swinging bass       → no extra bias (model handles it naturally).
        swing_bias = _compute_swing_bias(features, stage)

        return {
            "mode":                mode,
            "seed":                seed,
            "n_notes":             n_notes,
            "temperature":         temperature,
            "contour_target":      contour_target,
            "chord_idx":           chord_idx,
            "scale_pitch_classes": scale_pcs,
            "swing_bias":          swing_bias,
            "stage":               stage,
            "leadership":          self._leadership,
            "harmonic_mode":       self._harmony.current_mode_name(),
        }


# -----------------------------------------------------------------------
# Stage parameter tables
# -----------------------------------------------------------------------

def _stage_base_length(stage: str) -> int:
    return {
        "sparse":         5,
        "building":       9,
        "peak":          14,
        "recapitulation": 9,
        "resolution":     4,
    }.get(stage, 6)


def _stage_temperature(stage: str) -> float:
    return {
        "sparse":         0.80,
        "building":       0.90,
        "peak":           1.05,
        "recapitulation": 0.85,
        "resolution":     0.70,
    }.get(stage, 0.90)


def _compute_swing_bias(features: dict, stage: str) -> float:
    """
    Decide how strongly to bias sax duration tokens toward the triplet grid.

    Logic:
      - Straight bass feel  → strong triplet bias (rhythmic contrast)
      - Swinging bass feel  → no extra bias (LSTM already learned swing)
      - Mixed/unknown       → light bias as default swing flavour
      - Peak stage          → always some bias (maximum rhythmic interest)
    """
    feel = features.get("rhythmic_feel", "mixed")

    if stage == "peak":
        return 0.7   # always push triplet feel at peak intensity

    if feel == "straight":
        return 1.0   # maximum contrast: straight call → triplet response
    elif feel == "swing":
        return 0.0   # already swinging; let the model do its thing
    else:
        return 0.3   # light default swing flavour


def _should_recall(stage: str, memory: PhraseMemory) -> bool:
    recall_prob = {
        "sparse":         0.05,
        "building":       0.25,
        "peak":           0.15,
        "recapitulation": 0.70,
        "resolution":     0.40,
    }.get(stage, 0.1)
    has_memory = bool(memory.recall_recent("bass", n=1))
    return has_memory and random.random() < recall_prob

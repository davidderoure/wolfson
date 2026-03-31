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
from input.phrase_analyzer import analyze, complement_contour, complement_energy_arc
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

    def reset(self):
        """Reset arc timing and harmonic state for a new loop iteration.
        Memory is cleared separately by the caller (PhraseMemory.reset()).
        """
        self._start_time         = None
        self._last_bass_time     = None
        self._last_sax_time      = None
        self._last_bass_features = None
        self._leadership         = "bass"
        self._harmony            = HarmonyController()
        self._last_harmonic_stage = None

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


        # ------------------------------------------------------------------
        # Bass pitch-class tracking: steer the sax toward the tonality the
        # bassist is actually implying, rather than following the arc's
        # harmonic plan blindly.
        #
        # ≥ 4 distinct pitch classes → bassist is clearly implying a scale
        #   or mode; override the arc's scale bias with their pitch classes.
        # 2–3 pitch classes → a motif or interval fragment; broaden the arc
        #   scale by unioning in the bass notes so the sax stays compatible.
        # 0–1 pitch classes → single pedal tone or empty; defer to the arc.
        # ------------------------------------------------------------------
        bass_pcs = features.get("bass_pitch_classes", frozenset())
        MIN_PCS_FOR_OVERRIDE = 4

        if len(bass_pcs) >= MIN_PCS_FOR_OVERRIDE:
            scale_pcs    = bass_pcs        # sax follows bassist's harmonic language
            scale_source = "bass"
        elif len(bass_pcs) >= 2 and scale_pcs:
            scale_pcs    = scale_pcs | bass_pcs   # broaden: arc + bass notes
            scale_source = "blend"
        else:
            scale_source = "arc"           # arc harmony only

        # ------------------------------------------------------------------
        # Energy arc — internal phrase shaping
        # Complement the bass phrase's energy profile; override at key stages
        # (resolution always winds down; peak always arches).
        # ------------------------------------------------------------------
        phrase_energy_arc = complement_energy_arc(features)
        stage_arc_override = {"resolution": "ramp_down", "peak": "arch"}.get(stage)
        if stage_arc_override:
            phrase_energy_arc = stage_arc_override

        # ------------------------------------------------------------------
        # Motivic development — find recurring interval patterns in memory
        # Only use motifs seen at least twice; strength scales with stage.
        #
        # Lyrical motifs (extracted from sustained sax notes) are blended in
        # during quiet stages (recap, resolution) so the sax quotes back its
        # own singable lines rather than just any interval fragment.
        # ------------------------------------------------------------------
        motif_counter     = self.memory.recall_motifs(source=None, n_recent=16)
        motif_targets     = [m for m, cnt in motif_counter.most_common(2) if cnt >= 2]
        motif_strength    = _stage_motif_strength(stage)

        lyrical_strength  = _stage_lyrical_motif_strength(stage)
        if lyrical_strength > 0.0:
            lyrical_counter = self.memory.recall_lyrical_motifs(source="sax", n_recent=16)
            lyrical_targets = [m for m, cnt in lyrical_counter.most_common(2) if cnt >= 2]
            if lyrical_targets:
                # Put the top lyrical motif first; fill remaining slot from
                # the general pool (so we don't lose all non-lyrical development)
                motif_targets  = lyrical_targets[:1] + [
                    m for m in motif_targets if m not in lyrical_targets
                ][:1]
                motif_strength = max(motif_strength, lyrical_strength)
        modal_strength    = _stage_modal_strength(stage)

        # Rhythmic complementarity: blend the arc's stage density with the
        # reactive complement of the bass phrase's note density.
        # Dense bass (lots of notes/sec) → sparser sax; sparse bass → busier sax.
        # A 40% blend keeps the arc's macro shape in control while the sax
        # still reacts meaningfully to what the bassist is actually playing.
        REACTIVE_BLEND    = 0.4
        arc_density       = _stage_rhythmic_density(stage)
        bass_density_norm = min(1.0, features.get("note_density", 4.0) / 8.0)
        reactive_density  = 1.0 - bass_density_norm
        rhythmic_density  = (arc_density * (1.0 - REACTIVE_BLEND)
                             + reactive_density * REACTIVE_BLEND)

        # Register contrast: steer the sax toward the opposite register from
        # the bass, so call and response occupy different tonal spaces.
        # Modulated by ambitus: when the bass covers a wide range there is
        # less headroom for a clean contrast, so the effect is reduced.
        stage_contrast        = _stage_register_contrast(stage)
        bass_ambitus          = features.get("ambitus", 12)
        ambitus_factor        = max(0.0, 1.0 - bass_ambitus / 24.0)
        register_contrast_str = stage_contrast * ambitus_factor
        register_avoid_midi   = features.get("mean_pitch", 60.0)

        contour_target = complement_contour(features)

        # Phrase length: longer when sax leads, shorter when following.
        # Hard cap at 14 prevents runaway phrases; stage floor ensures even
        # the shortest "following" phrases are long enough to be musical.
        sax_leads   = (self._leadership == "sax") or proactive
        base_len    = _stage_base_length(stage)
        n_notes     = int(base_len * (1.1 if sax_leads else 0.75))
        n_notes_min = _stage_n_notes_floor(stage)
        n_notes     = max(n_notes_min, min(n_notes, 14))

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

        velocity = _compute_velocity(features, stage)

        return {
            "mode":                mode,
            "seed":                seed,
            "n_notes":             n_notes,
            "temperature":         temperature,
            "contour_target":      contour_target,
            "chord_idx":           chord_idx,
            "scale_pitch_classes": scale_pcs,
            "swing_bias":          swing_bias,
            "velocity":            velocity,
            "stage":               stage,
            "leadership":          self._leadership,
            "harmonic_mode":       self._harmony.current_mode_name(),
            "scale_source":        scale_source,
            "phrase_energy_arc":   phrase_energy_arc,
            "motif_targets":       motif_targets,
            "motif_strength":      motif_strength,
            "modal_strength":      modal_strength,
            "rhythmic_density":        rhythmic_density,
            "register_avoid_midi":     register_avoid_midi,
            "register_contrast_str":   register_contrast_str,
        }


# -----------------------------------------------------------------------
# Stage parameter tables
# -----------------------------------------------------------------------

def _stage_base_length(stage: str) -> int:
    return {
        "sparse":         5,
        "building":       8,   # was 9
        "peak":          12,   # was 14
        "recapitulation": 8,   # was 9
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


def _stage_swing_range(stage: str) -> tuple:
    """
    (min_swing, max_swing) band for each arc stage.

    The reactive feel-detection result is clamped to this band so that:

      sparse       — exploratory / rubato; never hard swing regardless of
                     what the detector reads.  The sax can sound almost
                     straight here, with just a hint of triplet colour.
      building     — groove establishing; always at least a light swing,
                     never locked all the way into full bebop triplets yet.
      peak         — hard swing; bypassed by the hard-coded 0.7 return.
      recapitulation — medium swing; recalls the feel without the bebop
                     intensity of the peak.
      resolution   — ballad lilt; gentle triplet colour, never busy swing.

    The band also breaks the self-play oscillation: lyrical long notes look
    "straight" to the detector → reactive=1.0; clamping to the stage ceiling
    stops that from forcing hard swing at quiet stages.
    """
    return {
        "sparse":          (0.0,  0.25),
        "building":        (0.3,  0.65),
        "peak":            (0.7,  0.7),   # hard-coded below; range unused
        "recapitulation":  (0.3,  0.60),
        "resolution":      (0.1,  0.30),
    }.get(stage, (0.2, 0.5))


def _compute_swing_bias(features: dict, stage: str) -> float:
    """
    Decide how strongly to bias sax duration tokens toward the triplet grid.

    Reactive component (from detected bass feel):
      straight → 1.0   maximum contrast: straight call → triplet response
      swing    → 0.0   let the stage range floor govern
      mixed    → 0.3   light default swing flavour

    The reactive value is then clamped to the stage's (min, max) band
    (see _stage_swing_range).  This gives each stage a distinct swing
    character that no detected feel can override:

      sparse      0.00–0.25   exploratory, barely swung
      building    0.30–0.65   groove warming up
      peak        0.70        always hard bebop swing
      recap       0.30–0.60   medium swing recall
      resolution  0.10–0.30   gentle ballad lilt
    """
    if stage == "peak":
        return 0.7   # hard-coded: peak is always hard swing

    feel = features.get("rhythmic_feel", "mixed")
    if feel == "straight":
        reactive = 1.0
    elif feel == "swing":
        reactive = 0.0
    else:
        reactive = 0.3

    lo, hi = _stage_swing_range(stage)
    return min(hi, max(lo, reactive))


def _compute_velocity(features: dict, stage: str) -> int:
    """
    Map bass input dynamics to sax output velocity.

    Strategy: mirror the bassist's mean velocity, scaled into a comfortable
    sax range (45-95), then apply a stage multiplier so the piece naturally
    breathes louder at peak and softer at the edges.

    Soft playing  (~32) → sax ~52   (pp/mp)
    Medium (~64)        → sax ~70   (mf)
    Loud  (~100)        → sax ~84   (f)
    Very loud (~120)    → sax ~92   (ff)
    """
    mean_vel = features.get("mean_velocity", 64)

    # Linear map: 0-127 input → 45-95 sax range
    base = int(45 + (mean_vel / 127.0) * 50)

    stage_scale = {
        "sparse":         0.85,   # hushed, exploratory
        "building":       1.00,
        "peak":           1.15,   # allow louder at climax
        "recapitulation": 0.95,
        "resolution":     0.80,   # fading out
    }.get(stage, 1.0)

    return max(40, min(110, int(base * stage_scale)))


def _stage_n_notes_floor(stage: str) -> int:
    """
    Minimum phrase length (notes) at each stage.

    Ensures that even the shortest "following" phrase (base_len × 0.75) is
    long enough to be musically meaningful once the singable-duration bias
    pulls individual notes toward quarter-note lengths.
    """
    return {
        "sparse":          3,
        "building":        6,
        "peak":            8,
        "recapitulation":  6,
        "resolution":      4,
    }.get(stage, 4)


def _stage_rhythmic_density(stage: str) -> float:
    """
    How busy/fast the sax should feel at each stage (0.0 = lyrical, 1.0 = bebop).

    Passed to PhraseGenerator as `rhythmic_density`; the singable-duration bias
    is scaled by (1 − rhythmic_density), so density=0 gives the full quarter-note
    pull and density=1 suppresses it entirely.

    Sparse and resolution are the most lyrical; peak is the busiest.
    """
    return {
        "sparse":          0.2,
        "building":        0.5,
        "peak":            0.9,
        "recapitulation":  0.3,
        "resolution":      0.1,
    }.get(stage, 0.5)


def _stage_register_contrast(stage: str) -> float:
    """
    Base strength of register-contrast bias at each arc stage.

    0.0 = no bias (sax can land anywhere)
    1.0 = strong push toward the register opposite the bass

    Sparse (dialogue barely begun) and peak (registers intentionally collide
    for maximum density) get low values.  Building, recapitulation and
    resolution — where call-and-response clarity matters most — get higher
    values.  The returned strength is further modulated by the bass ambitus
    in _build_params, so wide-range bass phrases reduce the effect.
    """
    return {
        "sparse":          0.0,
        "building":        0.5,
        "peak":            0.3,
        "recapitulation":  0.6,
        "resolution":      0.4,
    }.get(stage, 0.0)


def _stage_modal_strength(stage: str) -> float:
    """
    How strongly to boost P4/P5 leaps at each stage.

    Zero during tonal stages (sparse, resolution) where chromatic guide-tone
    motion is appropriate.  Rises through building; peaks at the modal climax.
    Recapitulation retains a modest lift as the harmonic language quotes back
    the modal material without fully committing to it.
    """
    return {
        "sparse":          0.0,
        "building":        0.6,
        "peak":            1.0,
        "recapitulation":  0.4,
        "resolution":      0.0,
    }.get(stage, 0.0)


def _stage_motif_strength(stage: str) -> float:
    """
    How strongly to bias toward recognised interval motifs at each stage.

    Zero in the sparse stage (not enough material yet).
    Grows through building and peak.
    Strongest in recapitulation — where thematic return is most meaningful.
    """
    return {
        "sparse":          0.0,
        "building":        0.3,
        "peak":            0.6,
        "recapitulation":  0.8,
        "resolution":      0.4,
    }.get(stage, 0.0)


def _stage_lyrical_motif_strength(stage: str) -> float:
    """
    How strongly to bias toward *lyrical* (sustained-note) sax motifs.

    Zero during the busy peak — those motifs are ornamental fragments.
    Strong in recapitulation and resolution, where the sax should quote back
    the singable themes it built earlier, creating a sense of returning home.
    Building gets a small lift so lyrical seeds planted early start to
    re-emerge before the recapitulation makes them explicit.
    """
    return {
        "sparse":          0.0,
        "building":        0.2,
        "peak":            0.0,
        "recapitulation":  0.7,
        "resolution":      0.5,
    }.get(stage, 0.0)


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

"""
Wolfson — interactive jazz bass + sax improvisation system.

Pipeline:
  MidiListener → PhraseDetector  → PhraseAnalyzer
                      │
                 BeatEstimator   (tap tempo from bass onsets)
                      │
                 ArcController   (5-min arc, leadership, proactive mode)
                      │
                 PhraseGenerator (LSTM + contour steering + chord conditioning)
                      │
                 MidiOutput      (per-note duration, articulation)

Proactive thread: fires every PROACTIVE_CHECK_INTERVAL seconds; if the arc
controller decides the sax should initiate, it generates and plays a phrase
without waiting for a bass phrase to complete.

Self-play mode (--self-play):
  The system feeds its own sax output back as input, creating an autonomous
  generative loop. No MIDI hardware needed. A short seed phrase bootstraps
  the first exchange; thereafter the sax continuously responds to itself.
  The 5-minute structural arc still governs the performance.
"""

import argparse
import random
import threading
import time
from collections import Counter, deque

from data.chords             import chord_index_to_name
from input.midi_listener    import MidiListener
from input.phrase_detector  import PhraseDetector
from input.phrase_analyzer  import analyze, extract_interval_motifs, extract_lyrical_motifs
from input.beat_estimator   import BeatEstimator
from memory.phrase_memory   import PhraseMemory
from generator.phrase_generator import PhraseGenerator, MAX_PHRASE_BEATS
from controller.arc_controller  import ArcController
from output.midi_output     import MidiOutput
from output.dashboard       import WolfsonDashboard
from output.osc_output      import OscOutput
from output.web_display     import WebAudienceDisplay
from config import (
    DEFAULT_INSTRUMENT, TEMPO_HINT_BPM,
    DASHBOARD_ENABLED, OSC_ENABLED, OSC_HOST, OSC_PORT,
    ARC, REST_PITCH,
    SELF_PLAY_CH_A, SELF_PLAY_CH_B,
    TRADE_BEATS_MODE, TRADE_BEATS_MIN,
    TINYURL_TOKEN, TINYURL_ALIAS,
    CLOUDFLARE_TUNNEL_NAME, CLOUDFLARE_TUNNEL_HOSTNAME,
    THIN_THRESHOLD_BEATS,
)

# Total arc duration in seconds (end of the last stage)
ARC_DURATION_SEC = max(end for _, end in ARC.values())

PROACTIVE_CHECK_INTERVAL  = 0.5   # seconds between proactive checks
RIFF_EVOLVE_THRESHOLD     = 2     # consecutive bass repeats before sax shifts to evolve mode
                                  # (value of 2 means: original + 2 repeats = 3rd occurrence)
SAX_RIFF_EVOLVE_THRESHOLD = 2     # consecutive sax repeats before sax shifts to develop mode

# Phrase statistics summary
STATS_WINDOW = 8    # rolling window — how many recent phrases are counted
STATS_EVERY  = 8    # print a summary block after every N phrases

# Self-play: brief silence between phrases (seconds) — musical breathing room
SELF_PLAY_PHRASE_GAP = 0.05

# Self-play seed: D minor pentatonic opening motif (pitch, beat-duration pairs)
# Gives the LSTM something musical to respond to on the very first exchange.
_SEED_PITCHES    = [62, 65, 67, 69, 72, 69, 67]   # D4 F4 G4 A4 C5 A4 G4
_SEED_DUR_BEATS  = [0.5, 0.5, 0.5, 0.5, 1.0, 0.5, 1.0]


class _PhraseStats:
    """
    Rolling-window statistics for phrase generation parameters.

    Records the last STATS_WINDOW phrases and prints a compact summary
    block every STATS_EVERY phrases — useful for monitoring harmonic
    mode distribution, scale tracking, and arc progress at a glance.
    """

    def __init__(self, window: int = STATS_WINDOW):
        self._total   = 0
        self._harm    = deque(maxlen=window)
        self._scale   = deque(maxlen=window)
        self._contour = deque(maxlen=window)
        self._stage   = deque(maxlen=window)
        self._arc     = deque(maxlen=window)

    def record(self, params: dict):
        self._total += 1
        self._harm.append(params.get("harmonic_mode", "?"))
        self._scale.append(params.get("scale_source",  "arc"))
        self._contour.append(params.get("contour_target", "?"))
        self._stage.append(params.get("stage", "?"))
        self._arc.append(params.get("phrase_energy_arc", "flat"))

    def should_print(self) -> bool:
        return self._total > 0 and self._total % STATS_EVERY == 0

    def print_summary(self, elapsed_sec: float):
        mins  = int(elapsed_sec // 60)
        secs  = int(elapsed_sec % 60)
        n     = len(self._harm)
        width = 60

        def fmt(d: deque) -> str:
            return "  ".join(f"{k}:{v}" for k, v in Counter(d).most_common())

        header = f"── last {n} phrases  t={mins}:{secs:02d}  phrase #{self._total} "
        print(header.ljust(width, "─"))
        print(f"  harm    {fmt(self._harm)}")
        print(f"  scale   {fmt(self._scale)}")
        print(f"  arc     {fmt(self._arc)}")
        print(f"  contour {fmt(self._contour)}")
        print(f"  stage   {fmt(self._stage)}")
        print("─" * width)


def main():
    parser = argparse.ArgumentParser(
        description="Wolfson interactive jazz improvisation system"
    )
    parser.add_argument(
        "--self-play", action="store_true",
        help="Autonomous mode: sax feeds its output back as its own input. "
             "No MIDI hardware required.",
    )
    parser.add_argument(
        "--bpm", type=float, default=120.0,
        help="Tempo for self-play mode (default: 120)",
    )
    parser.add_argument(
        "--dashboard", action="store_true", default=DASHBOARD_ENABLED,
        help="Enable full-screen rich terminal dashboard",
    )
    parser.add_argument(
        "--osc-host", metavar="HOST", default=None,
        help="Enable OSC output to HOST (e.g. 127.0.0.1 or 192.168.1.10)",
    )
    parser.add_argument(
        "--osc-port", type=int, default=OSC_PORT,
        help=f"OSC UDP port (default: {OSC_PORT})",
    )
    parser.add_argument(
        "--web", action="store_true", default=False,
        help="Serve an audience display page on the local network (default port 5000). "
             "Audience connect via http://<your-ip>:5000 on the same WiFi.",
    )
    parser.add_argument(
        "--web-port", type=int, default=5000,
        help="Port for the audience web display (default: 5000)",
    )
    parser.add_argument(
        "--tunnel", action="store_true", default=False,
        help="Open a cloudflared public tunnel so audience can connect from any network "
             "(e.g. eduroam).  Requires: brew install cloudflare/cloudflare/cloudflared",
    )
    parser.add_argument(
        "--trade", action="store_true", default=TRADE_BEATS_MODE,
        help="Beat-matching mode: sax response matches the bass phrase length "
             "in beats, enabling natural trading of 2s, 4s, or 8s.",
    )
    parser.add_argument(
        "--loop", action="store_true", default=False,
        help="Loop continuously: after each 5-minute arc completes, clear "
             "memory and start a new arc automatically. Useful for "
             "installations with multiple users. Ctrl-C to stop.",
    )
    parser.add_argument(
        "--loop-gap", type=float, default=8.0,
        help="Seconds to pause between arc loops (default: 8).",
    )
    parser.add_argument(
        "--riff-prob", type=float, default=0.0,
        metavar="P",
        help="Self-play: probability (0–1) that the bass re-plays its previous "
             "phrase instead of using the latest sax output. Simulates a "
             "repeating bass riff or ostinato. After 3+ consecutive repeats "
             "the sax boosts motivic development and adds contour direction. "
             "(default: 0.0 = always use latest sax output)",
    )
    parser.add_argument(
        "--sax-riff-prob", type=float, default=0.0,
        metavar="P",
        help="Probability (0–1) that the sax replays its previous phrase "
             "instead of generating a new response — the reverse of --riff-prob. "
             "The sax insists on a phrase while the bassist is free to develop "
             "underneath. After SAX_RIFF_EVOLVE_THRESHOLD consecutive repeats "
             "the sax shifts to development mode: it generates a variation of "
             "its repeated phrase with boosted motivic strength and directed "
             "contour, rather than looping indefinitely. "
             "(default: 0.0 = always generate a fresh response)",
    )
    parser.add_argument(
        "--chord-hint", action="store_true",
        help="Play a short voiced chord on a separate MIDI channel each time "
             "the harmony changes, making the internal harmonic state directly "
             "audible. Like a pianist lightly comping the chord at the start "
             "of each exchange. Use --comp-channel to set the MIDI channel.",
    )
    parser.add_argument(
        "--comp-channel", type=int, default=3,
        metavar="CH",
        help="MIDI channel for chord hint playback (default: 3). "
             "Route this to a piano or pad voice in your DAW.",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0,
        metavar="T",
        help="Offset added to the arc's per-stage generation temperature "
             "(default: 0.0 = arc defaults unchanged). Positive values make "
             "the sax more adventurous and unpredictable; negative values make "
             "it more conservative and idiomatic. Stage defaults range from "
             "0.70 (resolution) to 1.05 (peak); practical range for this "
             "offset is roughly -0.3 to +0.3.",
    )
    parser.add_argument(
        "--auto-start", action="store_true", default=False,
        help="Start the arc immediately on launch so the sax plays proactively "
             "before the bassist plays anything. Without this flag the arc starts "
             "lazily on the first bass phrase. Use this when you want Wolfson to "
             "open the performance and have the bassist join in.",
    )
    args = parser.parse_args()

    self_play      = args.self_play
    initial_bpm    = args.bpm
    use_dashboard  = args.dashboard
    osc_host       = args.osc_host    # None = OSC disabled
    trade_mode     = args.trade
    use_web        = args.web
    chord_hint     = args.chord_hint
    comp_channel   = args.comp_channel
    temp_offset    = args.temperature

    memory    = PhraseMemory()
    generator = PhraseGenerator(instrument=DEFAULT_INSTRUMENT)
    arc       = ArcController(memory)
    midi_out  = MidiOutput()
    beats     = BeatEstimator(
        initial_bpm = initial_bpm,
        hint_bpm    = TEMPO_HINT_BPM or initial_bpm,
    )

    _running  = threading.Event()
    _running.set()
    _sax_lock = threading.Lock()   # one sax phrase at a time
    _stats    = _PhraseStats()

    # Self-play two-channel dialogue.
    # Phrases alternate between SELF_PLAY_CH_A and SELF_PLAY_CH_B so a DAW
    # can route them to separate sounds (e.g. alto sax / tenor sax), making
    # the call-and-response structure directly audible.
    # Stored as a one-element list so the closure can mutate it.
    _sp_parity = [0]   # 0 → CH_A, 1 → CH_B; flipped after every phrase

    riff_prob     = args.riff_prob if self_play else 0.0
    sax_riff_prob = args.sax_riff_prob

    # Riff / ostinato tracking (self-play only).
    # _last_injected_bass_notes: the notes most recently fed to on_bass_phrase
    #   via the self-play loop — stored so they can be re-injected on a riff.
    # _last_bass_pitches / _bass_repeat_count: detect consecutive repetitions
    #   so the sax can respond differently once the bass is clearly looping.
    _last_injected_bass_notes = [None]
    _last_bass_pitches        = [None]   # tuple of pitches from previous bass phrase
    _bass_repeat_count        = [0]      # consecutive identical-phrase count

    # Sax riff tracking (--sax-riff-prob).
    # _last_sax_notes: the most recently generated sax phrase (beat durations),
    #   stored so it can be replayed verbatim on a sax riff cycle.
    # _sax_repeat_count: consecutive sax replays; resets when a fresh phrase
    #   is generated.
    _last_sax_notes   = [None]
    _sax_repeat_count = [0]

    def _phrase_pitches(phrase: list[dict]) -> tuple:
        """Return a tuple of pitched note MIDI values for riff comparison."""
        return tuple(n["pitch"] for n in phrase
                     if n.get("pitch", REST_PITCH) != REST_PITCH)

    def _phrase_matches(a: tuple, b: tuple) -> bool:
        """
        Fuzzy phrase identity test for riff/repeat detection.

        Two phrases are considered the same riff if:
          - Their lengths differ by at most 1 note (handles the occasional
            extra grace note or missed note at a phrase boundary).
          - At least (n - 1) of the first n pitch classes match in order,
            where n = length of the shorter phrase.  One mismatch is
            allowed to accommodate a note played in a different octave or
            a single wrong note.

        Pitch class (MIDI % 12) is used rather than absolute MIDI pitch so
        that the same riff played an octave up or down is still recognised.
        """
        if abs(len(a) - len(b)) > 1:
            return False
        a_pc = tuple(p % 12 for p in a)
        b_pc = tuple(p % 12 for p in b)
        n = min(len(a_pc), len(b_pc))
        if n == 0:
            return False
        matches = sum(x == y for x, y in zip(a_pc[:n], b_pc[:n]))
        return matches >= n - 1   # tolerate one mismatch

    dashboard  = WolfsonDashboard() if use_dashboard else None
    osc_out    = OscOutput(osc_host, args.osc_port) if osc_host else None
    web_out    = WebAudienceDisplay(port=args.web_port) if use_web else None

    # ------------------------------------------------------------------
    # Bass phrase handler (reactive path)
    # ------------------------------------------------------------------

    # Arc is started lazily on the first bass phrase so the web server
    # and cloudflared tunnel can be running (and the waiting screen shown)
    # for as long as needed before the performance begins.
    _arc_started = threading.Event()

    def on_bass_phrase(phrase: list[dict]):
        if not _arc_started.is_set():
            arc.start()
            _arc_started.set()
        # Annotate notes with current beat duration so summary stats can
        # compute beat-relative durations for bass phrases (which arrive from
        # PhraseDetector in wall-clock seconds, not beats).
        bds = beats.beat_duration
        for n in phrase:
            if "beat_dur_sec" not in n:
                n["beat_dur_sec"] = bds

        # Track consecutive bass phrase repetitions.
        # After RIFF_EVOLVE_THRESHOLD repeats the sax shifts from trading
        # to development mode — stronger motif use and directed contour.
        pitches = _phrase_pitches(phrase)
        if _last_bass_pitches[0] is not None and _phrase_matches(pitches, _last_bass_pitches[0]):
            _bass_repeat_count[0] += 1
        else:
            _bass_repeat_count[0] = 0
        _last_bass_pitches[0] = pitches

        motifs = extract_interval_motifs(phrase)
        memory.store(phrase, source="bass", motifs=motifs)
        params = arc.on_bass_phrase(phrase)

        # Riff evolution: once the bass has repeated the same phrase
        # RIFF_EVOLVE_THRESHOLD times in a row, boost the sax's motivic
        # development and push contour away from neutral so the response
        # clearly evolves rather than trading the same lick back.
        repeat = _bass_repeat_count[0]
        if repeat >= RIFF_EVOLVE_THRESHOLD:
            boost = min(0.4, 0.15 * (repeat - RIFF_EVOLVE_THRESHOLD + 1))
            params["motif_strength"] = min(1.0,
                params.get("motif_strength", 0.0) + boost)
            if params.get("contour_target", "neutral") == "neutral":
                params["contour_target"] = (
                    "ascending" if repeat % 2 == 0 else "descending")
            if not dashboard:
                print(f"  [riff ×{repeat + 1}  motif_str→{params['motif_strength']:.2f}"
                      f"  contour→{params['contour_target']}]")

        if trade_mode and len(phrase) >= 2:
            # Measure the bass phrase in beats and cap the sax response to
            # the same length — enabling natural trading of 2s, 4s, or 8s.
            phrase_dur_sec   = phrase[-1]["offset"] - phrase[0]["onset"]
            phrase_beats     = phrase_dur_sec / beats.beat_duration
            params["max_phrase_beats"] = max(TRADE_BEATS_MIN, phrase_beats)

        _respond(params, triggered_by="bass")

    # ------------------------------------------------------------------
    # Shared response path
    # ------------------------------------------------------------------

    def _respond(params: dict, triggered_by: str):
        if params is None:
            return
        if not _sax_lock.acquire(blocking=False):
            return   # sax already mid-phrase

        notes_out    = []
        beat_dur_sec = beats.beat_duration

        # In self-play, alternate between the two dialogue channels so each
        # "voice" stays on its own MIDI channel throughout the performance.
        # In live-bass mode, always use CH_A (channel 1).
        out_channel = (
            (SELF_PLAY_CH_A if _sp_parity[0] == 0 else SELF_PLAY_CH_B)
            if self_play else SELF_PLAY_CH_A
        )

        # Sax riff: decide whether to replay the last phrase or generate fresh.
        # With probability sax_riff_prob, replay the stored phrase verbatim
        # (up to SAX_RIFF_EVOLVE_THRESHOLD times).  Once the threshold is
        # reached, generate a variation instead — boosted motif strength and
        # directed contour — so the sax is heard to develop its insistence
        # rather than loop indefinitely.  Reset the counter after any fresh
        # generation.
        _sax_replay = (
            sax_riff_prob > 0.0
            and _last_sax_notes[0] is not None
            and random.random() < sax_riff_prob
            and _sax_repeat_count[0] < SAX_RIFF_EVOLVE_THRESHOLD
        )

        if _sax_replay:
            _sax_repeat_count[0] += 1
            if not dashboard:
                print(f"  [sax riff ×{_sax_repeat_count[0]}]")
        else:
            if _sax_repeat_count[0] >= SAX_RIFF_EVOLVE_THRESHOLD:
                # Development mode: vary the repeated phrase rather than
                # abandoning it — mirrors what happens on the bass riff path.
                boost = min(0.4, 0.15 * (_sax_repeat_count[0]
                                         - SAX_RIFF_EVOLVE_THRESHOLD + 1))
                params["motif_strength"] = min(
                    1.0, params.get("motif_strength", 0.0) + boost)
                if params.get("contour_target", "neutral") == "neutral":
                    params["contour_target"] = (
                        "ascending" if _sax_repeat_count[0] % 2 == 0
                        else "descending")
                if not dashboard:
                    print(f"  [sax riff ×{_sax_repeat_count[0]} develop"
                          f"  motif_str→{params['motif_strength']:.2f}"
                          f"  contour→{params['contour_target']}]")
            _sax_repeat_count[0] = 0

        try:
            if _sax_replay:
                notes = list(_last_sax_notes[0])
            else:
                notes = generator.generate(
                    seed_phrase         = params["seed"],
                    tempo_bpm           = beats.bpm,
                    n_notes             = params["n_notes"],
                    temperature         = max(0.1, params["temperature"] + temp_offset),
                    contour_target      = params["contour_target"],
                    chord_idx           = params["chord_idx"],
                    swing_bias          = params.get("swing_bias", 0.0),
                    scale_pitch_classes = params.get("scale_pitch_classes"),
                    phrase_energy_arc   = params.get("phrase_energy_arc", "flat"),
                    motif_targets       = params.get("motif_targets", []),
                    motif_strength      = params.get("motif_strength", 0.0),
                    modal_strength      = params.get("modal_strength", 0.0),
                    rhythmic_density      = params.get("rhythmic_density", 0.5),
                    max_phrase_beats      = params.get("max_phrase_beats", MAX_PHRASE_BEATS),
                    register_avoid_midi   = params.get("register_avoid_midi", 60.0),
                    register_contrast_str = params.get("register_contrast_str", 0.0),
                )
            if not notes:
                # Model generated nothing (END_TOKEN sampled immediately).
                # In self-play, re-inject the seed so the loop doesn't stall.
                if self_play and _running.is_set():
                    beat_dur_sec = beats.beat_duration
                    _schedule_feedback(
                        [{"pitch":          n["pitch"],
                          "duration_beats": (n["offset"] - n["onset"])
                                            / beat_dur_sec}
                         for n in params["seed"]],
                        beat_dur_sec,
                    )
                return

            beat_dur_sec  = beats.beat_duration
            durations_sec = [n["duration_beats"] * beat_dur_sec for n in notes]

            # Build a properly-timed sax phrase for memory storage.
            # Rest sentinels advance the clock but are not stored as pitched notes.
            _t = 0.0
            sax_phrase = []
            base_vel   = params.get("velocity", 80)
            for n in notes:
                dur_sec = n["duration_beats"] * beat_dur_sec
                if n["pitch"] != REST_PITCH:
                    sax_phrase.append({
                        "pitch":        n["pitch"],
                        "velocity":     80,
                        "onset":        _t,
                        "offset":       _t + dur_sec,
                        "beat_dur_sec": beat_dur_sec,
                    })
                _t += dur_sec
            sax_motifs         = extract_interval_motifs(notes)
            sax_lyrical_motifs = extract_lyrical_motifs(notes)
            memory.store(sax_phrase, source="sax",
                         motifs=sax_motifs, lyrical_motifs=sax_lyrical_motifs)

            # Update displays and send OSC before playback so receivers
            # can react in sync with the first note.
            if chord_hint and midi_out:
                midi_out.play_chord_hint(
                    params["chord_idx"],
                    beats.beat_duration,
                    channel = comp_channel,
                )
            if dashboard:
                dashboard.update(params, notes, beats.bpm,
                                 arc.elapsed(), triggered_by)
            else:
                _log(params, triggered_by, notes, beats.bpm, out_channel)

            if web_out:
                web_out.update(params, notes, beats.bpm,
                               arc.elapsed(), triggered_by)

            if osc_out:
                osc_out.send_phrase(params, notes, beats.bpm,
                                    arc.elapsed(), triggered_by)

            # Stochastic performance thinning — drop some short notes at
            # output only.  Memory and motif detection already used the full
            # phrase above, so internal musical intelligence is unaffected.
            played_notes = _thin_phrase(notes, stage=params.get("stage", "building"))
            played_durs  = [n["duration_beats"] * beat_dur_sec for n in played_notes]
            played_vel   = _shape_phrase_dynamics(played_notes, base_vel)
            midi_out.play_phrase(
                pitches   = [n["pitch"] for n in played_notes],
                durations = played_durs,
                velocity  = played_vel,
                channel   = out_channel,
            )

            notes_out = notes   # capture after successful play
            _last_sax_notes[0] = notes  # store for potential sax riff replay

            # Stats: always recorded; summary printed only in text mode
            _stats.record(params)
            if not dashboard and _stats.should_print():
                _stats.print_summary(arc.elapsed())

        finally:
            arc.on_sax_played()
            if self_play:
                _sp_parity[0] ^= 1   # flip 0↔1 for next phrase
            _sax_lock.release()

        # Self-play feedback — lock already released before this runs
        if self_play and notes_out and _running.is_set():
            _schedule_feedback(notes_out, beat_dur_sec)

    # ------------------------------------------------------------------
    # Self-play feedback
    # ------------------------------------------------------------------

    def _schedule_feedback(notes: list[dict], beat_dur_sec: float):
        """
        Inject sax notes back as a bass phrase in a new daemon thread.
        A short gap simulates the silence between phrases.

        With --riff-prob > 0 the feedback occasionally re-injects the
        *previous* bass phrase instead of the latest sax output, simulating
        a repeating bass riff or ostinato.  The repetition counter in
        on_bass_phrase() then detects how many times the same phrase has
        appeared and boosts the sax's motivic development accordingly.
        """
        import random as _rnd

        # Decide which notes to inject before spawning the thread so the
        # random choice is made at scheduling time, not after the sleep.
        source_notes = notes
        if riff_prob > 0.0 and _last_injected_bass_notes[0] is not None:
            if _rnd.random() < riff_prob:
                source_notes = _last_injected_bass_notes[0]

        # Record what is actually being injected (for the next riff decision).
        _last_injected_bass_notes[0] = source_notes

        def _run():
            time.sleep(SELF_PLAY_PHRASE_GAP)
            if not _running.is_set():
                return
            now    = time.time()
            phrase = []
            onset  = 0.0
            for n in source_notes:
                dur_sec = n["duration_beats"] * beat_dur_sec
                if n["pitch"] != REST_PITCH:
                    phrase.append({
                        "pitch":        n["pitch"],
                        "velocity":     64,
                        "onset":        now + onset,
                        "offset":       now + onset + dur_sec,
                        "beat_dur_sec": beat_dur_sec,
                    })
                # Do NOT call beats.note_on() in self-play — the generated
                # note IOIs are at the generation tempo, not the live-input
                # tempo, and would cause the estimator to runaway to 300+ BPM.
                onset += dur_sec
            if phrase:
                on_bass_phrase(phrase)

        threading.Thread(target=_run, daemon=True).start()

    # ------------------------------------------------------------------
    # Proactive background thread
    # ------------------------------------------------------------------

    _summary_computed = [False]
    _summary_shown_at = [None]   # wall time when summary was pushed to web
    loop_mode = args.loop
    loop_gap  = args.loop_gap

    def _restart_arc():
        """Reset state for a new loop iteration and re-seed self-play."""
        _summary_computed[0] = False
        _arc_started.clear()
        memory.reset()
        arc.reset()
        _last_injected_bass_notes[0] = None
        _last_bass_pitches[0]        = None
        _bass_repeat_count[0]        = 0
        _last_sax_notes[0]           = None
        _sax_repeat_count[0]         = 0
        if web_out:
            web_out.reset_summary()
        if not dashboard:
            print(f"\nStarting new arc. Gap: {loop_gap:.0f}s.\n")
        if self_play:
            def _bootstrap_loop():
                time.sleep(loop_gap)
                if not _running.is_set():
                    return
                beat_dur = 60.0 / initial_bpm
                now      = time.time()
                phrase   = []
                onset    = 0.0
                for p, d in zip(_SEED_PITCHES, _SEED_DUR_BEATS):
                    dur_sec = d * beat_dur
                    phrase.append({
                        "pitch":    p,
                        "velocity": 64,
                        "onset":    now + onset,
                        "offset":   now + onset + dur_sec,
                    })
                    beats.note_on(now + onset)
                    onset += dur_sec
                on_bass_phrase(phrase)
            threading.Thread(target=_bootstrap_loop, daemon=True).start()

    def _proactive_loop():
        while _running.is_set():
            time.sleep(PROACTIVE_CHECK_INTERVAL)
            elapsed = arc.elapsed()
            # Arc completion: compute performance summary once
            if elapsed >= ARC_DURATION_SEC and not _summary_computed[0]:
                _summary_computed[0] = True
                summary = _compute_performance_summary(memory)
                _print_performance_summary(summary)
                if web_out:
                    web_out.show_summary(summary)
                    _summary_shown_at[0] = time.time()
                if loop_mode:
                    _restart_arc()
                    continue
            # Auto-stop self-play at arc completion (300s) when not looping.
            # When the web display is active, wait at least 6 seconds after
            # show_summary() so the browser has several poll cycles to pick up
            # the summary before the loop exits.
            if self_play and not loop_mode and elapsed >= ARC_DURATION_SEC:
                shown_at = _summary_shown_at[0]
                if shown_at is not None and time.time() - shown_at < 6.0:
                    pass   # keep looping until grace period expires
                else:
                    if not dashboard:
                        print("\nArc complete. Stopping.")
                    _running.clear()
                    break
            if elapsed < ARC_DURATION_SEC and arc.should_play_proactively():
                params = arc.get_proactive_params()
                if params:
                    _respond(params, triggered_by="sax")

    # ------------------------------------------------------------------
    # MIDI I/O and note-on hook for beat estimation
    # ------------------------------------------------------------------

    def _note_on_with_beat(pitch, velocity, t):
        beats.note_on(t)
        detector.note_on(pitch, velocity, t)

    detector = PhraseDetector(on_phrase_complete=on_bass_phrase)
    listener = MidiListener(
        on_note_on  = _note_on_with_beat,
        on_note_off = detector.note_off,
    )

    midi_out.start()
    midi_out.silence([SELF_PLAY_CH_A, SELF_PLAY_CH_B])   # clear any notes from previous run

    if dashboard:
        dashboard.start()

    if web_out:
        web_out.start(
            tunnel        = args.tunnel,
            tinyurl_token = TINYURL_TOKEN,
            tinyurl_alias = TINYURL_ALIAS,
            tunnel_name   = CLOUDFLARE_TUNNEL_NAME,
            tunnel_host   = CLOUDFLARE_TUNNEL_HOSTNAME,
        )

    if osc_out:
        print(f"OSC output → {osc_out}")

    if trade_mode and not dashboard:
        print(f"Beat-matching (--trade) enabled: sax phrases will match bass phrase length.")

    if self_play:
        # Seed the loop with an opening phrase in a background thread
        # so main() is not blocked before the KeyboardInterrupt handler.
        def _bootstrap():
            time.sleep(0.2)   # brief pause to allow MIDI output to settle
            if not _running.is_set():
                return
            beat_dur = 60.0 / initial_bpm
            now      = time.time()
            phrase   = []
            onset    = 0.0
            for p, d in zip(_SEED_PITCHES, _SEED_DUR_BEATS):
                dur_sec = d * beat_dur
                phrase.append({
                    "pitch":    p,
                    "velocity": 64,
                    "onset":    now + onset,
                    "offset":   now + onset + dur_sec,
                })
                beats.note_on(now + onset)
                onset += dur_sec
            on_bass_phrase(phrase)

        threading.Thread(target=_bootstrap, daemon=True).start()
        if not dashboard:
            print(
                f"Wolfson self-play mode. {initial_bpm:.0f} BPM. "
                "Ctrl-C to stop.\n"
            )
    else:
        listener.start()
        if args.auto_start:
            arc.start()
            _arc_started.set()
            if not dashboard:
                print("Wolfson auto-start: arc running. Sax will play proactively.\n"
                      "Join in whenever you're ready.\n")
        elif not dashboard:
            print("Wolfson ready. Play bass. Ctrl-C to stop.\n")

    proactive_thread = threading.Thread(target=_proactive_loop, daemon=True)
    proactive_thread.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        if not dashboard:
            print("\nStopping.")
    finally:
        _running.clear()
        if dashboard:
            dashboard.stop()
        if web_out:
            web_out.stop()
        if not self_play:
            listener.stop()
        midi_out.silence([SELF_PLAY_CH_A, SELF_PLAY_CH_B])
        midi_out.stop()


# ------------------------------------------------------------------
# Performance summary
# ------------------------------------------------------------------

_SUMMARY_NOTE_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def _midi_to_name(midi: int) -> str:
    return _SUMMARY_NOTE_NAMES[midi % 12] + str(midi // 12 - 1)


def _compute_performance_summary(memory) -> dict:
    """
    Compute two-column stats from PhraseMemory for the end-of-arc display.

    Returns a dict with keys "bass", "sax", and "observations", where
    bass/sax are stat dicts and observations is a list of plain-text
    observations addressed to the human player.
    """
    def stats(source: str) -> dict:
        entries = memory._filter(source)
        if not entries:
            return {"phrases": 0}

        dur_beats_all = []
        velocities    = []
        pitches       = []

        for entry in entries:
            for n in entry["phrase"]:
                p = n.get("pitch", REST_PITCH)
                if p == REST_PITCH or p <= 0:
                    continue
                bds = n.get("beat_dur_sec", 0.0)
                if bds > 0:
                    dur_beats_all.append((n["offset"] - n["onset"]) / bds)
                velocities.append(n.get("velocity", 64))
                pitches.append(p)

        result: dict = {"phrases": len(entries), "notes": len(pitches)}

        if dur_beats_all:
            mean_dur  = sum(dur_beats_all) / len(dur_beats_all)
            short_pct = 100 * sum(d < 0.4 for d in dur_beats_all) / len(dur_beats_all)
            result["mean_dur"]  = round(mean_dur, 2)
            result["short_pct"] = int(round(short_pct))

        if pitches:
            result["pitch_lo"] = _midi_to_name(min(pitches))
            result["pitch_hi"] = _midi_to_name(max(pitches))

        if velocities:
            result["vel_lo"] = min(velocities)
            result["vel_hi"] = max(velocities)

        return result

    bass = stats("bass")
    sax  = stats("sax")

    # Plain-text observations addressed to the human player — only fired when
    # a value is clearly outside a reasonable range.  Framed as observations,
    # not judgements.
    observations: list[str] = []
    if "mean_dur" in bass:
        if bass["mean_dur"] < 0.40:
            observations.append(
                "Your notes were quite short — try sustaining longer ideas.")
        elif bass["mean_dur"] > 1.20:
            observations.append(
                "You played long, sustained notes — good melodic space.")
    if "vel_lo" in bass and "vel_hi" in bass:
        if bass["vel_hi"] - bass["vel_lo"] < 20:
            observations.append(
                "Your dynamic range was narrow — try varying your touch more.")
    if bass.get("phrases", 0) > 0 and sax.get("phrases", 0) > 0:
        ratio = bass["phrases"] / sax["phrases"]
        if ratio < 0.5:
            observations.append(
                "The sax played many more phrases than you — "
                "try being more active in the conversation.")
        elif ratio > 2.0:
            observations.append(
                "You drove most of the conversation — "
                "try leaving more space for the sax to initiate.")

    return {"bass": bass, "sax": sax, "observations": observations}


def _print_performance_summary(summary: dict):
    """Print the two-column performance summary to the console."""
    bass = summary.get("bass", {})
    sax  = summary.get("sax",  {})
    obs  = summary.get("observations", [])
    w    = 54

    def row(label, bval, sval):
        print(f"  {label:<18} {str(bval):>12}  {str(sval):>12}")

    print("\n" + "═" * w)
    print("  PERFORMANCE SUMMARY")
    print("═" * w)
    row("", "BASS (you)", "SAX")
    print("  " + "─" * (w - 2))
    row("Phrases",     bass.get("phrases", "—"), sax.get("phrases", "—"))
    row("Notes",       bass.get("notes",   "—"), sax.get("notes",   "—"))
    if "mean_dur" in bass or "mean_dur" in sax:
        row("Note length",
            f"{bass['mean_dur']:.2f}b" if "mean_dur" in bass else "—",
            f"{sax['mean_dur']:.2f}b"  if "mean_dur" in sax  else "—")
    if "short_pct" in bass or "short_pct" in sax:
        row("Short notes",
            f"{bass['short_pct']}%" if "short_pct" in bass else "—",
            f"{sax['short_pct']}%"  if "short_pct" in sax  else "—")
    if "pitch_lo" in bass or "pitch_lo" in sax:
        row("Pitch range",
            f"{bass['pitch_lo']} – {bass['pitch_hi']}" if "pitch_lo" in bass else "—",
            f"{sax['pitch_lo']} – {sax['pitch_hi']}"   if "pitch_lo" in sax  else "—")
    if "vel_lo" in bass or "vel_lo" in sax:
        row("Dynamics",
            f"{bass['vel_lo']} – {bass['vel_hi']}" if "vel_lo" in bass else "—",
            f"{sax['vel_lo']} – {sax['vel_hi']}"   if "vel_lo" in sax  else "—")
    print("═" * w)
    if obs:
        print()
        for o in obs:
            print(f"  — {o}")
    print()


# ------------------------------------------------------------------
# Stochastic performance thinning
# ------------------------------------------------------------------

# Per-stage maximum drop probability for eligible (short) notes.
# Zero at sparse — every note is musically precious when material is thin.
# Rises through building/peak where fast runs benefit most from thinning.
# Gentle at recap/resolution where lyrical long notes dominate anyway.
_THIN_STAGE_STRENGTH = {
    "sparse":          0.00,
    "building":        0.10,
    "peak":            0.15,
    "recapitulation":  0.08,
    "resolution":      0.05,
}


def _thin_phrase(notes: list, stage: str) -> list:
    """
    Return a thinned copy of *notes* for MIDI output only.

    The full phrase is already stored in memory before this is called, so
    motif detection, arc feedback, and self-play seeding are unaffected.

    Rules:
      - REST_PITCH sentinels are always kept (they carry timing).
      - The first and last *pitched* notes are always kept.
      - Notes with duration_beats >= THIN_THRESHOLD_BEATS are immune.
      - Eligible notes are dropped with probability that scales linearly
        from 0 at the threshold down to `stage_strength` at zero duration:
            p = stage_strength * (1 - dur / THIN_THRESHOLD_BEATS)
    """
    import random as _random

    if THIN_THRESHOLD_BEATS <= 0.0:
        return notes

    strength = _THIN_STAGE_STRENGTH.get(stage, 0.0)
    if strength == 0.0:
        return notes

    # Identify indices of pitched notes so we can protect first and last
    pitched_indices = [i for i, n in enumerate(notes) if n.get("pitch") != REST_PITCH]
    if len(pitched_indices) < 3:
        return notes   # phrase too short to thin meaningfully

    protected = {pitched_indices[0], pitched_indices[-1]}

    out = []
    for i, note in enumerate(notes):
        if note.get("pitch") == REST_PITCH or i in protected:
            out.append(note)
            continue

        dur = note.get("duration_beats", 1.0)
        if dur >= THIN_THRESHOLD_BEATS:
            out.append(note)
            continue

        # Linear drop probability: maximum at dur→0, zero at threshold
        p_drop = strength * (1.0 - dur / THIN_THRESHOLD_BEATS)
        if _random.random() >= p_drop:
            out.append(note)
        # else: note silently omitted from MIDI output

    return out


# ------------------------------------------------------------------
# Phrase-shape dynamics
# ------------------------------------------------------------------

# Velocity multiplier applied to the highest-pitch note in a phrase.
_PEAK_ACCENT_FACTOR = 1.15

# Velocity multipliers for the penultimate and last pitched notes.
# Simulates breath pressure dropping at the end of a phrase.
_TAPER_END_FACTORS = [0.85, 0.70]


def _shape_phrase_dynamics(notes: list, base_vel: int) -> list[int]:
    """
    Compute per-note MIDI velocities with melodic peak accent and end taper.

    1. Peak accent  — the highest-pitched note gets a +15% velocity boost,
       reflecting the natural jazz tendency to push dynamically on the peak.
    2. End taper    — the last two pitched notes are reduced to 85% / 70% of
       their computed velocity, simulating breath pressure dropping at the
       phrase end and giving each phrase a sense of release.

    Both passes layer on top of the energy-arc velocity_scale that is already
    encoded per-note by the generator, so they augment rather than replace the
    existing dynamic shape.

    REST_PITCH sentinels are given a nominal velocity of 0 and are never played.
    """
    pitched = [(i, n) for i, n in enumerate(notes) if n.get("pitch") != REST_PITCH]

    # Base velocity shaped by the energy-arc velocity_scale
    vels = [
        max(40, min(110, int(base_vel * n.get("velocity_scale", 1.0))))
        if n.get("pitch") != REST_PITCH else 0
        for n in notes
    ]

    if len(pitched) < 3:
        return vels   # phrase too short to shape meaningfully

    # 1. Melodic peak accent
    peak_idx = max(pitched, key=lambda x: x[1].get("pitch", 0))[0]
    vels[peak_idx] = min(120, int(vels[peak_idx] * _PEAK_ACCENT_FACTOR))

    # 2. Phrase-end taper — last two pitched notes
    for slot, (i, _) in enumerate(pitched[-2:]):
        vels[i] = max(30, int(vels[i] * _TAPER_END_FACTORS[slot]))

    return vels


# ------------------------------------------------------------------
# Console logging
# ------------------------------------------------------------------

def _log(params: dict, triggered_by: str, notes: list, bpm: float,
         channel: int = 1):
    stage   = params.get("stage", "?")
    lead    = params.get("leadership", "?")
    mode    = params.get("mode", "?")
    contour = params.get("contour_target", "?")
    harm    = params.get("harmonic_mode", "?")
    src     = params.get("scale_source", "arc")
    arc     = params.get("phrase_energy_arc", "flat")
    motifs  = len(params.get("motif_targets", []))
    vel     = params.get("velocity", 80)
    modal   = params.get("modal_strength", 0.0)
    density = params.get("rhythmic_density", 0.5)
    mpb     = params.get("max_phrase_beats")
    mpb_str = f"  cap={mpb:.1f}b" if mpb is not None else ""
    chord   = chord_index_to_name(params.get("chord_idx", 48))
    print(
        f"[{stage:>14s}]  {bpm:5.1f} bpm  ch={channel}  lead={lead:<3s}  "
        f"trigger={triggered_by:<4s}  mode={mode:<8s}  "
        f"chord={chord:<5s}  harm={harm:<12s}  scale={src:<5s}  arc={arc:<9s}  "
        f"motifs={motifs}  contour={contour:<10s}  vel={vel:3d}  "
        f"modal={modal:.1f}  density={density:.1f}  n={len(notes)}{mpb_str}"
    )


if __name__ == "__main__":
    main()

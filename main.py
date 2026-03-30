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
import threading
import time
from collections import Counter, deque

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
)

# Total arc duration in seconds (end of the last stage)
ARC_DURATION_SEC = max(end for _, end in ARC.values())

PROACTIVE_CHECK_INTERVAL = 0.5   # seconds between proactive checks

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
        "--trade", action="store_true", default=TRADE_BEATS_MODE,
        help="Beat-matching mode: sax response matches the bass phrase length "
             "in beats, enabling natural trading of 2s, 4s, or 8s.",
    )
    args = parser.parse_args()

    self_play      = args.self_play
    initial_bpm    = args.bpm
    use_dashboard  = args.dashboard
    osc_host       = args.osc_host    # None = OSC disabled
    trade_mode     = args.trade
    use_web        = args.web

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

    dashboard  = WolfsonDashboard() if use_dashboard else None
    osc_out    = OscOutput(osc_host, args.osc_port) if osc_host else None
    web_out    = WebAudienceDisplay(port=args.web_port) if use_web else None

    # ------------------------------------------------------------------
    # Bass phrase handler (reactive path)
    # ------------------------------------------------------------------

    def on_bass_phrase(phrase: list[dict]):
        motifs = extract_interval_motifs(phrase)
        memory.store(phrase, source="bass", motifs=motifs)
        params = arc.on_bass_phrase(phrase)
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

        try:
            notes = generator.generate(
                seed_phrase         = params["seed"],
                tempo_bpm           = beats.bpm,
                n_notes             = params["n_notes"],
                temperature         = params["temperature"],
                contour_target      = params["contour_target"],
                chord_idx           = params["chord_idx"],
                swing_bias          = params.get("swing_bias", 0.0),
                scale_pitch_classes = params.get("scale_pitch_classes"),
                phrase_energy_arc   = params.get("phrase_energy_arc", "flat"),
                motif_targets       = params.get("motif_targets", []),
                motif_strength      = params.get("motif_strength", 0.0),
                modal_strength      = params.get("modal_strength", 0.0),
                rhythmic_density    = params.get("rhythmic_density", 0.5),
                max_phrase_beats    = params.get("max_phrase_beats", MAX_PHRASE_BEATS),
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

            # Per-note velocities from energy arc (velocity_scale 0.75–1.25)
            per_note_vel = [
                max(40, min(110, int(base_vel * n.get("velocity_scale", 1.0))))
                for n in notes
            ]
            midi_out.play_phrase(
                pitches   = [n["pitch"] for n in notes],
                durations = durations_sec,
                velocity  = per_note_vel,
                channel   = out_channel,
            )

            notes_out = notes   # capture after successful play

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
        """
        def _run():
            time.sleep(SELF_PLAY_PHRASE_GAP)
            if not _running.is_set():
                return
            now    = time.time()
            phrase = []
            onset  = 0.0
            for n in notes:
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

    def _proactive_loop():
        while _running.is_set():
            time.sleep(PROACTIVE_CHECK_INTERVAL)
            # Auto-stop self-play at arc completion (300s)
            if self_play and arc.elapsed() >= ARC_DURATION_SEC:
                print("\nArc complete. Stopping.")
                _running.clear()
                break
            if arc.should_play_proactively():
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
    arc.start()

    if dashboard:
        dashboard.start()

    if web_out:
        web_out.start()

    if osc_out:
        print(f"OSC output → {osc_out}")

    if trade_mode and not dashboard:
        print(f"Beat-matching (--trade) enabled: sax phrases will match bass phrase length.")

    if self_play:
        # Seed the loop with an opening phrase in a background thread
        # so main() is not blocked before the KeyboardInterrupt handler.
        def _bootstrap():
            time.sleep(0.2)   # allow arc to initialise
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
        if not dashboard:
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
    print(
        f"[{stage:>14s}]  {bpm:5.1f} bpm  ch={channel}  lead={lead:<3s}  "
        f"trigger={triggered_by:<4s}  mode={mode:<8s}  "
        f"harm={harm:<12s}  scale={src:<5s}  arc={arc:<9s}  "
        f"motifs={motifs}  contour={contour:<10s}  vel={vel:3d}  "
        f"modal={modal:.1f}  density={density:.1f}  n={len(notes)}{mpb_str}"
    )


if __name__ == "__main__":
    main()

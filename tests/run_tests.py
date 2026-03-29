"""
Wolfson test suite.

Runs each test case programmatically (no MIDI hardware required).
Writes a per-test log file and a combined two-track demo MIDI.

Usage:
    cd ~/wolfson
    python tests/run_tests.py

Outputs:
    tests/logs/01_basic_response.txt  ...  tests/logs/14_tempo_200bpm.txt
    tests/demo.mid
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pretty_midi

from generator.phrase_generator import PhraseGenerator
from input.phrase_analyzer import analyze, complement_contour
from data.chords import parse_chord, chord_index_to_name, NC_INDEX
from data.scales import scale_pitch_classes, chord_to_mode, chord_root
from config import GENERATION_TEMPERATURE

LOGS_DIR  = Path(__file__).parent / "logs"
DEMO_MIDI = Path(__file__).parent / "demo.mid"

DEFAULT_BPM = 120.0
GAP_BEATS   = 2           # silence between segments in demo MIDI (in beats)

MIDI_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
ROOT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def midi_name(pitch: int) -> str:
    return f"{MIDI_NAMES[pitch % 12]}{(pitch // 12) - 1}"


# ---------------------------------------------------------------------------
# Phrase builder — bpm-aware
# ---------------------------------------------------------------------------

def make_phrase(pitches, beat_durs, velocity=80, bpm=DEFAULT_BPM):
    """
    Build a phrase list from pitches and beat-relative durations at a given BPM.
    IOI between notes equals beat_dur in seconds.
    """
    beat_dur = 60.0 / bpm
    notes = []
    t = 0.0
    for pitch, beats in zip(pitches, beat_durs):
        dur_sec = beats * beat_dur
        notes.append({
            "pitch":        pitch,
            "onset":        t,
            "offset":       t + dur_sec * 0.85,
            "velocity":     velocity,
            "beat_dur_sec": beat_dur,
        })
        t += dur_sec
    return notes


def _scale_label(params: dict) -> str:
    spc = params.get("scale_pitch_classes")
    if spc is None:
        return "chromatic"
    chord_idx = params.get("chord_idx", NC_INDEX)
    root = chord_root(chord_idx)
    mode = chord_to_mode(chord_idx)
    return f"{ROOT_NAMES[root]} {mode}"


# ---------------------------------------------------------------------------
# Single-test runner — bpm-aware
# ---------------------------------------------------------------------------

def run_test(generator, title, bass_phrase, params, description="", bpm=DEFAULT_BPM):
    """
    Analyse the bass phrase, generate a sax response.
    Returns (log_text, response, features, sax_velocity).
    """
    beat_dur = 60.0 / bpm
    features = analyze(bass_phrase)

    if "contour_target" not in params:
        params["contour_target"] = complement_contour(features)
    if "swing_bias" not in params:
        params["swing_bias"] = 1.0 if features["rhythmic_feel"] == "straight" else 0.0

    response = generator.generate(
        seed_phrase          = bass_phrase,
        tempo_bpm            = bpm,
        n_notes              = params.get("n_notes", 8),
        temperature          = params.get("temperature", GENERATION_TEMPERATURE),
        contour_target       = params.get("contour_target", "neutral"),
        chord_idx            = params.get("chord_idx", NC_INDEX),
        swing_bias           = params.get("swing_bias", 0.0),
        scale_pitch_classes  = params.get("scale_pitch_classes"),
    )

    sax_velocity = min(110, max(40, features["mean_velocity"]))

    # Sax duration in seconds at this tempo
    sax_dur_s = [f"{n['duration_beats'] * beat_dur:.2f}" for n in response]

    lines = []
    if description:
        lines += [f"Description: {description}", ""]

    bass_dur_sec = bass_phrase[-1]["offset"] - bass_phrase[0]["onset"]
    lines += [
        "Bass phrase:",
        f"  Pitches:    {' '.join(midi_name(n['pitch']) for n in bass_phrase)}",
        f"  Tempo:      {bpm:.0f} BPM",
        f"  Vel:        {bass_phrase[0]['velocity']}",
        f"  Duration:   {bass_dur_sec:.2f}s",
        "",
        "Analysis:",
        f"  note_density:    {features['note_density']:.2f} notes/sec",
        f"  ambitus:         {features['ambitus']} semitones",
        f"  contour_slope:   {features['contour_slope']:+.2f}",
        f"  rhetorical_type: {features['rhetorical_type']}",
        f"  rhythmic_feel:   {features['rhythmic_feel']} (ratio={features['swing_ratio']:.2f})",
        f"  mean_velocity:   {features['mean_velocity']}",
        f"  is_sparse:       {features['is_sparse']}",
        "",
        "Generation parameters:",
        f"  chord:           {chord_index_to_name(params.get('chord_idx', NC_INDEX))}",
        f"  scale:           {_scale_label(params)}",
        f"  contour_target:  {params.get('contour_target', 'neutral')}",
        f"  swing_bias:      {params.get('swing_bias', 0.0):.2f}",
        f"  temperature:     {params.get('temperature', GENERATION_TEMPERATURE):.2f}",
        f"  sax_velocity:    {sax_velocity}",
        "",
        "Sax response:",
        f"  n notes:  {len(response)}",
        f"  Pitches:  {' '.join(midi_name(n['pitch']) for n in response)}",
        "  Dur (b):  " + ' '.join(f"{n['duration_beats']:.2f}" for n in response),
        "  Dur (s):  " + ' '.join(sax_dur_s),
    ]

    return "\n".join(lines), response, features, sax_velocity


# ---------------------------------------------------------------------------
# Demo MIDI builder — per-segment BPM
# ---------------------------------------------------------------------------

def build_demo_midi(segments):
    """
    Build a two-track MIDI: bass (track 0) + sax (track 1).

    segments: list of (bass_phrase, sax_response, sax_velocity, label, bpm)

    Note timings are in wall-clock seconds so each segment plays at its own
    tempo regardless of the MIDI file's global tempo marker.
    """
    midi     = pretty_midi.PrettyMIDI(initial_tempo=DEFAULT_BPM)
    bass_ins = pretty_midi.Instrument(program=33, name="Bass")
    sax_ins  = pretty_midi.Instrument(program=65, name="Alto Sax")

    cursor = 60.0 / DEFAULT_BPM * 2   # 2-beat lead-in at default tempo

    for bass_phrase, sax_response, sax_velocity, _label, bpm in segments:
        beat_dur     = 60.0 / bpm
        phrase_start = cursor

        # Bass notes
        for n in bass_phrase:
            start = phrase_start + (n["onset"] - bass_phrase[0]["onset"])
            end   = phrase_start + (n["offset"] - bass_phrase[0]["onset"])
            bass_ins.notes.append(pretty_midi.Note(
                velocity=n.get("velocity", 80), pitch=n["pitch"],
                start=start, end=max(end, start + 0.05),
            ))

        bass_dur = bass_phrase[-1]["offset"] - bass_phrase[0]["onset"]
        cursor   = phrase_start + bass_dur + beat_dur   # 1-beat gap to sax

        # Sax notes
        sax_start = cursor
        for n in sax_response:
            dur_sec = n["duration_beats"] * beat_dur
            end_sec = sax_start + dur_sec * 0.85
            sax_ins.notes.append(pretty_midi.Note(
                velocity=sax_velocity, pitch=n["pitch"],
                start=sax_start, end=max(end_sec, sax_start + 0.05),
            ))
            sax_start += dur_sec

        sax_dur = sum(n["duration_beats"] * beat_dur for n in sax_response)
        cursor  = cursor + sax_dur + beat_dur * GAP_BEATS

    midi.instruments.append(bass_ins)
    midi.instruments.append(sax_ins)
    midi.write(str(DEMO_MIDI))


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------

def define_tests():
    Dm7  = parse_chord("Dm7")
    G7   = parse_chord("G7")
    Cmaj = parse_chord("C")

    # Shared ascending motif used by multiple tests
    MOTIF_PITCHES   = [52, 53, 55, 57, 60, 62]
    MOTIF_BEAT_DURS = [1,  1,  1,  1,  1,  1 ]

    return [
        # ── 1 ─────────────────────────────────────────────────────────────
        {
            "name":        "01_basic_response",
            "title":       "Basic call-response",
            "description": "A simple ascending phrase. Tests that the system responds musically.",
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 2 ─────────────────────────────────────────────────────────────
        {
            "name":        "02_dynamics_soft",
            "title":       "Dynamics — soft",
            "description": "Same phrase at low velocity (vel=30). Sax should respond softly.",
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    30,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 3 ─────────────────────────────────────────────────────────────
        {
            "name":        "03_dynamics_loud",
            "title":       "Dynamics — loud",
            "description": "Same phrase at high velocity (vel=100). Sax should respond loudly.",
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    100,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 4 ─────────────────────────────────────────────────────────────
        {
            "name":        "04_contour_question",
            "title":       "Contour — question phrase",
            "description": (
                "Ascending phrase ending on a high note. Should be classified as 'question'; "
                "sax responds with descending 'answer' contour."
            ),
            "pitches":     [48, 52, 55, 57, 60, 64],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 5 ─────────────────────────────────────────────────────────────
        {
            "name":        "05_contour_answer",
            "title":       "Contour — answer phrase",
            "description": (
                "Descending phrase ending on a low note. Should be classified as 'answer'; "
                "sax responds with ascending 'question' contour."
            ),
            "pitches":     [64, 60, 57, 55, 52, 48],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 6 ─────────────────────────────────────────────────────────────
        {
            "name":        "06_swing_feel",
            "title":       "Swing feel — detected, no extra bias",
            "description": (
                "Swung 8th notes (2:1 long-short ratio per beat). Swing ratio should be ~2.0; "
                "classified as 'swing'; no additional triplet bias applied to response."
            ),
            "pitches":     [57, 60, 57, 55, 52, 50, 52, 55],
            "beat_durs":   [2/3, 1/3, 2/3, 1/3, 2/3, 1/3, 2/3, 1/3],
            "velocity":    80,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 7 ─────────────────────────────────────────────────────────────
        {
            "name":        "07_straight_feel",
            "title":       "Straight feel — triplet bias in response",
            "description": (
                "Even quarter notes (straight feel). Swing ratio ~1.0; classified as 'straight'; "
                "sax response gets triplet-grid duration bias for rhythmic contrast."
            ),
            "pitches":     [57, 60, 57, 55, 52, 50, 52, 55],
            "beat_durs":   [1,  1,  1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
            "bpm":         120.0,
            "params":      {},
        },
        # ── 8 ─────────────────────────────────────────────────────────────
        {
            "name":        "08_d_dorian",
            "title":       "Modal — D Dorian scale bias",
            "description": (
                "Phrase over Dm7 with D Dorian scale pitch bias active. "
                "Sax should favour D, E, F, G, A, B, C scale tones."
            ),
            "pitches":     [50, 52, 53, 55, 57, 59, 60, 62],
            "beat_durs":   [1,  1,  1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
            "bpm":         120.0,
            "params":      {
                "chord_idx":           Dm7,
                "scale_pitch_classes": scale_pitch_classes(2, "dorian"),
            },
        },
        # ── 9 ─────────────────────────────────────────────────────────────
        {
            "name":        "09_ii_V_I",
            "title":       "ii-V-I progression",
            "description": (
                "Same bass phrase generated three times, each over a different chord: "
                "Dm7 (Dorian), G7 (Mixolydian), Cmaj (Ionian). "
                "Demonstrates chord-conditioned harmonic colour changing across the progression."
            ),
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         120.0,
            "params":      {
                "chord_idx":           Dm7,
                "scale_pitch_classes": scale_pitch_classes(2, "dorian"),
            },
            "progression": [
                {"chord_idx": G7,   "scale_pitch_classes": scale_pitch_classes(7, "mixolydian")},
                {"chord_idx": Cmaj, "scale_pitch_classes": scale_pitch_classes(0, "ionian")},
            ],
        },
        # ── 10 ────────────────────────────────────────────────────────────
        {
            "name":        "10_sparse",
            "title":       "Sparse phrase — sax fills space",
            "description": (
                "Two widely-spaced notes at low velocity. High sparsity score; "
                "sax responds with a longer, denser phrase to fill the space."
            ),
            "pitches":     [52, 60],
            "beat_durs":   [2,  2 ],
            "velocity":    55,
            "bpm":         120.0,
            "params":      {"n_notes": 12},
        },
        # ── 11–14: Tempo variation ─────────────────────────────────────────
        # Same ascending motif at four tempos. The token encoding is beat-
        # relative so phrase shape should be similar, but the LSTM may prefer
        # different note densities and durations at different tempos since the
        # training data spans a broad tempo range.
        {
            "name":        "11_tempo_60bpm",
            "title":       "Tempo — 60 BPM (ballad)",
            "description": (
                "Same ascending motif at 60 BPM. Each beat is 1.0s; the phrase "
                "spans 6s. Compare sax response shape and note density with tests "
                "01/12/13/14 to see if tempo influences phrase character."
            ),
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         60.0,
            "params":      {},
        },
        {
            "name":        "12_tempo_90bpm",
            "title":       "Tempo — 90 BPM (slow swing)",
            "description": (
                "Same ascending motif at 90 BPM. Each beat is 0.67s; the phrase "
                "spans 4s."
            ),
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         90.0,
            "params":      {},
        },
        {
            "name":        "13_tempo_160bpm",
            "title":       "Tempo — 160 BPM (up-tempo)",
            "description": (
                "Same ascending motif at 160 BPM. Each beat is 0.375s; the phrase "
                "spans 2.25s. At this speed the minimum duration floor (0.2 beats) "
                "may constrain the response."
            ),
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         160.0,
            "params":      {},
        },
        {
            "name":        "14_tempo_200bpm",
            "title":       "Tempo — 200 BPM (very fast)",
            "description": (
                "Same ascending motif at 200 BPM. Each beat is 0.3s; the phrase "
                "spans 1.8s. Tests how the system handles very fast tempos where "
                "the beat estimator may struggle with subdivision ambiguity."
            ),
            "pitches":     MOTIF_PITCHES,
            "beat_durs":   MOTIF_BEAT_DURS,
            "velocity":    80,
            "bpm":         200.0,
            "params":      {},
        },
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    generator = PhraseGenerator()
    print()

    tests    = define_tests()
    segments = []   # (bass_phrase, sax_response, sax_velocity, label, bpm)

    for test in tests:
        bpm = test.get("bpm", DEFAULT_BPM)
        print(f"  [{test['name']}]  {test['title']}")

        bass_phrase = make_phrase(test["pitches"], test["beat_durs"], test["velocity"], bpm)
        params      = test["params"].copy()

        log_text, response, features, sax_velocity = run_test(
            generator, test["title"], bass_phrase, params, test["description"], bpm
        )

        header = (
            "=" * 70 + "\n"
            f"  {test['title']}\n"
            + "=" * 70 + "\n\n"
        )

        # Handle progression sub-tests (ii-V-I)
        if "progression" in test:
            prog_lines = [header, log_text, ""]
            prog_lines.append("  — Dm7 response above —")
            prog_lines.append("")
            segments.append((bass_phrase, response, sax_velocity, f"{test['name']} Dm7", bpm))

            for prog_params_extra in test["progression"]:
                prog_params = params.copy()
                prog_params.update(prog_params_extra)
                prog_params.setdefault("contour_target", params.get("contour_target", "neutral"))
                prog_params.setdefault("swing_bias",     params.get("swing_bias",     0.0))

                _, prog_response, _, _ = run_test(
                    generator, test["title"], bass_phrase, prog_params, bpm=bpm
                )
                chord_name = chord_index_to_name(prog_params["chord_idx"])
                root = chord_root(prog_params["chord_idx"])
                mode = chord_to_mode(prog_params["chord_idx"])
                dur_str = ' '.join(f"{n['duration_beats']:.2f}" for n in prog_response)
                pit_str = ' '.join(midi_name(n['pitch']) for n in prog_response)
                prog_lines.append(
                    f"  Over {chord_name} ({ROOT_NAMES[root]} {mode}):\n"
                    f"  Pitches: {pit_str}\n"
                    f"  Dur(b):  {dur_str}\n"
                )
                segments.append((bass_phrase, prog_response, sax_velocity,
                                  f"{test['name']} {chord_name}", bpm))

            log_path = LOGS_DIR / f"{test['name']}.txt"
            log_path.write_text("\n".join(prog_lines))

        else:
            log_path = LOGS_DIR / f"{test['name']}.txt"
            log_path.write_text(header + log_text + "\n")
            segments.append((bass_phrase, response, sax_velocity, test["name"], bpm))

        print(f"         → {log_path}")

    print()
    build_demo_midi(segments)
    print(f"  → {DEMO_MIDI}")
    print(f"\nDone. {len(tests)} tests, {len(segments)} demo segments.")


if __name__ == "__main__":
    main()

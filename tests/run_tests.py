"""
Wolfson test suite.

Runs each test case programmatically (no MIDI hardware required).
Writes a per-test log file and a combined two-track demo MIDI.

Usage:
    cd ~/wolfson
    python tests/run_tests.py

Outputs:
    tests/logs/01_basic_response.txt
    tests/logs/02_dynamics_soft.txt
    ...
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

BPM      = 120.0
BEAT_DUR = 60.0 / BPM        # 0.5 s per beat at 120 BPM
GAP_SEC  = BEAT_DUR * 2      # silence between tests in demo MIDI


# ---------------------------------------------------------------------------
# Note/phrase helpers
# ---------------------------------------------------------------------------

MIDI_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
ROOT_NAMES = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']


def midi_name(pitch: int) -> str:
    return f"{MIDI_NAMES[pitch % 12]}{(pitch // 12) - 1}"


def make_phrase(pitches, beat_durs, velocity=80):
    """
    Build a phrase list from pitches and beat-relative durations.
    IOI between notes equals beat_dur * BEAT_DUR (legato-ish).
    """
    notes = []
    t = 0.0
    for pitch, beats in zip(pitches, beat_durs):
        dur_sec = beats * BEAT_DUR
        notes.append({
            "pitch":        pitch,
            "onset":        t,
            "offset":       t + dur_sec * 0.85,
            "velocity":     velocity,
            "beat_dur_sec": BEAT_DUR,
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
# Single-test runner
# ---------------------------------------------------------------------------

def run_test(generator, title, bass_phrase, params, description=""):
    """
    Analyse the bass phrase, generate a sax response, return (log_text, response).
    """
    features = analyze(bass_phrase)

    if "contour_target" not in params:
        params["contour_target"] = complement_contour(features)
    if "swing_bias" not in params:
        params["swing_bias"] = 1.0 if features["rhythmic_feel"] == "straight" else 0.0

    response = generator.generate(
        seed_phrase          = bass_phrase,
        tempo_bpm            = BPM,
        n_notes              = params.get("n_notes", 8),
        temperature          = params.get("temperature", GENERATION_TEMPERATURE),
        contour_target       = params.get("contour_target", "neutral"),
        chord_idx            = params.get("chord_idx", NC_INDEX),
        swing_bias           = params.get("swing_bias", 0.0),
        scale_pitch_classes  = params.get("scale_pitch_classes"),
    )

    sax_velocity = min(110, max(40, features["mean_velocity"]))

    lines = []
    if description:
        lines += [f"Description: {description}", ""]

    lines += [
        "Bass phrase:",
        f"  Pitches:    {' '.join(midi_name(n['pitch']) for n in bass_phrase)}",
        f"  Vel:        {bass_phrase[0]['velocity']}",
        f"  Duration:   {bass_phrase[-1]['offset'] - bass_phrase[0]['onset']:.2f}s",
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
    ]

    return "\n".join(lines), response, features, sax_velocity


# ---------------------------------------------------------------------------
# Demo MIDI builder
# ---------------------------------------------------------------------------

def build_demo_midi(segments):
    """
    Build a two-track MIDI: bass (track 0) + sax (track 1).

    segments: list of (bass_phrase, sax_response, velocity, label)
    """
    midi     = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    bass_ins = pretty_midi.Instrument(program=33, name="Bass")
    sax_ins  = pretty_midi.Instrument(program=65, name="Alto Sax")

    cursor = BEAT_DUR * 2   # 2-beat lead-in

    for bass_phrase, sax_response, sax_velocity, _label in segments:
        phrase_start = cursor

        # Write bass notes
        for n in bass_phrase:
            start = phrase_start + (n["onset"] - bass_phrase[0]["onset"])
            end   = phrase_start + (n["offset"] - bass_phrase[0]["onset"])
            bass_ins.notes.append(pretty_midi.Note(
                velocity=n.get("velocity", 80), pitch=n["pitch"],
                start=start, end=max(end, start + 0.05),
            ))

        bass_dur = bass_phrase[-1]["offset"] - bass_phrase[0]["onset"]
        cursor   = phrase_start + bass_dur + BEAT_DUR   # 1-beat gap to sax

        # Write sax notes
        sax_start = cursor
        for n in sax_response:
            dur_sec = n["duration_beats"] * BEAT_DUR
            end_sec = sax_start + dur_sec * 0.85
            sax_ins.notes.append(pretty_midi.Note(
                velocity=sax_velocity, pitch=n["pitch"],
                start=sax_start, end=max(end_sec, sax_start + 0.05),
            ))
            sax_start += dur_sec

        sax_dur = sum(n["duration_beats"] * BEAT_DUR for n in sax_response)
        cursor  = cursor + sax_dur + GAP_SEC

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

    return [
        # ── 1 ─────────────────────────────────────────────────────────────
        {
            "name":        "01_basic_response",
            "title":       "Basic call-response",
            "description": "A simple ascending phrase. Tests that the system responds musically.",
            "pitches":     [52, 53, 55, 57, 60, 62],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
            "params":      {},
        },
        # ── 2 ─────────────────────────────────────────────────────────────
        {
            "name":        "02_dynamics_soft",
            "title":       "Dynamics — soft",
            "description": "Same phrase at low velocity (vel=30). Sax should respond softly.",
            "pitches":     [52, 53, 55, 57, 60, 62],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    30,
            "params":      {},
        },
        # ── 3 ─────────────────────────────────────────────────────────────
        {
            "name":        "03_dynamics_loud",
            "title":       "Dynamics — loud",
            "description": "Same phrase at high velocity (vel=100). Sax should respond loudly.",
            "pitches":     [52, 53, 55, 57, 60, 62],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    100,
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
            "params":      {},   # swing_bias auto-set from analysis
        },
        # ── 8 ─────────────────────────────────────────────────────────────
        {
            "name":        "08_d_dorian",
            "title":       "Modal — D Dorian scale bias",
            "description": (
                "Phrase over Dm7 with D Dorian scale pitch bias active. "
                "Sax should favour D, E, F, G, A, B, C scale tones."
            ),
            "pitches":     [50, 52, 53, 55, 57, 59, 60, 62],  # D E F G A B C D
            "beat_durs":   [1,  1,  1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
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
            "pitches":     [52, 53, 55, 57, 60, 62],
            "beat_durs":   [1,  1,  1,  1,  1,  1 ],
            "velocity":    80,
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
            "params":      {"n_notes": 12},
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
    segments = []   # (bass_phrase, sax_response, sax_velocity, label) for MIDI

    for test in tests:
        print(f"  [{test['name']}]  {test['title']}")

        bass_phrase = make_phrase(test["pitches"], test["beat_durs"], test["velocity"])
        params      = test["params"].copy()

        log_text, response, features, sax_velocity = run_test(
            generator, test["title"], bass_phrase, params, test["description"]
        )

        # Build log file
        header = (
            "=" * 70 + "\n"
            f"  {test['title']}\n"
            + "=" * 70 + "\n\n"
        )

        # Handle progression sub-tests (ii-V-I)
        if "progression" in test:
            prog_lines = [header, log_text, ""]
            prog_lines.append(f"  — Dm7 response above —")
            prog_lines.append("")
            segments.append((bass_phrase, response, sax_velocity, f"{test['name']} Dm7"))

            for prog_params_extra in test["progression"]:
                prog_params = params.copy()
                prog_params.update(prog_params_extra)
                # carry over contour/swing from first run
                prog_params.setdefault("contour_target", params.get("contour_target", "neutral"))
                prog_params.setdefault("swing_bias",     params.get("swing_bias",     0.0))

                _, prog_response, _, _ = run_test(
                    generator, test["title"], bass_phrase, prog_params
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
                                  f"{test['name']} {chord_name}"))

            log_path = LOGS_DIR / f"{test['name']}.txt"
            log_path.write_text("\n".join(prog_lines))

        else:
            log_path = LOGS_DIR / f"{test['name']}.txt"
            log_path.write_text(header + log_text + "\n")
            segments.append((bass_phrase, response, sax_velocity, test["name"]))

        print(f"         → {log_path}")

    # Build demo MIDI
    print()
    build_demo_midi(segments)
    print(f"  → {DEMO_MIDI}")
    print(f"\nDone. {len(tests)} tests, {len(segments)} demo segments.")


if __name__ == "__main__":
    main()

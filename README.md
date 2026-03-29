# Wolfson

An interactive jazz improvisation system. You play bass; it plays sax. It listens, responds, and develops a musical conversation with you over a 5-minute performance arc.

## Overview

Wolfson uses an LSTM trained on jazz solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) to generate melodic sax responses to live bass input. The system is designed for live performance: it detects phrases in your playing, generates a response, and manages a structural arc over the duration of the piece.

### Architecture

```
Bass (pitch-to-MIDI) ──► MidiListener ──► PhraseDetector ──► PhraseAnalyzer
                               │                                     │
                          BeatEstimator                        PhraseMemory
                         (live tempo)                       (stores both voices)
                               │                                     │
                               └──────────── ArcController ──────────┘
                                          (arc, leadership,
                                           proactive mode)
                                                 │
                                        HarmonyController
                                    (mode, progression, pedal,
                                     tritone substitution)
                                                 │
                                          PhraseGenerator
                                    (LSTM + chord conditioning
                                     + scale pitch bias
                                     + contour steering
                                     + swing/triplet bias)
                                                 │
                                           MidiOutput ──► Synth (sax voice)
```

### Musicality features

**Phrase analysis** — each bass phrase is characterised by note density, pitch ambitus, contour slope, and rhetorical type (question / answer / neutral). The arc controller uses these to shape its response.

**Question and answer** — the sax detects whether a bass phrase is a question (rising, open ending) or an answer (falling, resolving), and responds with the complement. This creates the classic jazz call-and-response dialogue.

**Leadership and role swapping** — the system tracks who is leading at each moment. Sparse bass playing (low density, small range) signals the bassist is comping; the sax takes the initiative. Dense melodic bass signals the sax should respond. Leadership shifts deliberately over the arc.

**Contour steering** — soft logit biases on pitch tokens guide the sax phrase toward a target contour (ascending / descending / neutral). Steering is applied from the first generated note using the seed phrase's mean pitch as the reference, so the whole arc of the response reflects the rhetorical intent rather than just the tail. Bias strength increases linearly with distance from the reference pitch.

**Chord conditioning** — the LSTM is trained with chord context from the WJD beats table (49-token chord vocabulary: 12 roots × 4 quality classes + NC). At runtime the `HarmonyController` issues a chord index each phrase; the model degrades gracefully to NC if no chord is supplied.

**Harmonic modes** — the `HarmonyController` operates in one of four modes, selected automatically by the performance stage:

| Mode | Behaviour |
|------|-----------|
| `free` | No harmonic steering; chromatic scale bias (sparse stage) |
| `modal` | Root + mode (e.g. D Dorian) held for N phrases, then drifts by a semitone; used in building and recapitulation |
| `progression` | Steps through a chord progression one chord per phrase — ii-V-I, VI-II-V-I, I-VI-II-V, or 12-bar blues; tritone substitution on V7 chords (~35%); used at peak |
| `pedal` | Fixed bass pedal tone with cycling upper harmony (i → bVII7 → i → V7); used in resolution |

**Scale pitch bias** — positive logit bias is added to pitch tokens whose pitch class belongs to the current chord's scale or mode. Non-scale tones are not penalised, so chromatic passing notes remain available. Bias strength is set to give clearly audible harmonic colour across different modes and chord qualities.

**Pitch range** — a soft logit penalty steers generated pitches into a practical sax register (E3–E6). Notes outside this range are penalised proportionally to their distance from the limit, preventing the generator from following the bass into an unplayable register while still allowing occasional extremes.

**Swing / triplet feel** — the system detects whether the bass is playing straight or swung (from consecutive IOI ratios). A straight bass call gets a triplet-grid duration bias in the response, pushing the sax toward the 12/8 feel and creating rhythmic contrast. A swinging bass gets no bias — the LSTM's learned distribution handles it.

**Live tempo tracking** — a `BeatEstimator` infers BPM from inter-onset intervals in the bass line. Generated sax phrases play back in time with the bassist's actual tempo. No click track or advance setup required.

**Dynamics** — the mean MIDI velocity of each bass phrase is tracked and mapped to the sax output velocity, so the sax mirrors the bassist's dynamic level. A stage multiplier modulates this further: sparse and resolution stages are inherently softer; the peak stage pushes louder. Playing quietly draws a quiet response; playing hard drives the sax to match.

**Articulation** — each generated note sounds for 85% of its time slot, with a short silence before the next note. A minimum duration floor (0.2 beats, ≈100 ms at 120 BPM) prevents imperceptibly short notes from appearing in dense generated passages.

**Proactive mode** — the sax does not always wait for a bass phrase to end. When the bassist is sparse or silent, the sax initiates. During the resolution stage, the sax always plays the final phrase.

### Performance arc

| Time | Stage | Harmonic mode | Character |
|------|-------|---------------|-----------|
| 0:00–1:00 | sparse | free | Short exchanges; bass leads; establish motifs |
| 1:00–2:30 | building | modal | Longer phrases; settle into a mode; sax begins recalling |
| 2:30–3:30 | peak | progression | Maximum density; active chord motion; tritone subs |
| 3:30–4:30 | recapitulation | modal | Return to early phrases; modal feel restored |
| 4:30–5:00 | resolution | pedal | Sax thins and resolves over a pedal tone; plays the last phrase |

## Training data

Solos from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) (WJD). The saxophone subset (alto, tenor, baritone, soprano) contains 271 solos — ~106,000 notes across ~5,000 phrases. Chord changes are extracted per note from the WJD beats table.

The system supports multiple instrument families (trumpet, trombone, flute) with separate models.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `python-rtmidi`, `torch`, `numpy`, `pretty_midi`

### Hardware

- Monophonic pitch-to-MIDI converter on your instrument
- MIDI interface
- Synth or software instrument on a MIDI output channel for the sax voice

### MIDI configuration

Edit `config.py` to set your MIDI port indices:

```python
MIDI_INPUT_PORT = 0    # your pitch-to-MIDI interface
MIDI_OUTPUT_PORT = 0   # your sax synth
```

To list available ports:
```bash
python -c "import rtmidi; m=rtmidi.MidiIn(); print(m.get_ports())"
```

To verify that MIDI events are arriving on the correct input port before starting the main script:
```bash
python test-midi-in.py
```
Play a few notes — you should see `Note ON` and `Note OFF` lines for each. Adjust `MIDI_INPUT_PORT` in `config.py` until events appear, then run `main.py`.

## Training

### 1. Get the data

Download the Weimar Jazz Database from [jazzomat.hfm-weimar.de](https://jazzomat.hfm-weimar.de/download/download.html):

- `wjazzd.db` — SQLite database (solo metadata, note data, chord changes)
- `RELEASE2.0_mid_unquant.zip` — unquantised MIDI files (optional, for `--midi-only` mode)

Place `wjazzd.db` in `data/raw/`. Extract the zip to `data/raw/midi_unquant/` if using MIDI-only mode.

### 2. Prepare training data

```bash
# Inspect the database (instrument distribution, schema)
python data/prepare.py --inspect

# Extract saxophone phrases with chord sequences
python data/prepare.py --instrument sax
```

### 3. Train

```bash
python generator/train.py --instrument sax --epochs 100
```

Best model saved to `models/sax_best.pt`.

**On Google Colab:** open `wolfson_train.ipynb` — handles data upload, training on GPU, and model download.

### 4. Run

```bash
python main.py
```

## Testing individual features

`demo.py` lets you test any single feature in isolation without the 5-minute arc. Each bass phrase you play triggers one sax response using fixed, explicit parameters.

```bash
python demo.py --demo scale         # D Dorian scale pitch bias
python demo.py --demo swing         # triplet-grid response to your playing
python demo.py --demo straight      # no swing bias (compare with above)
python demo.py --demo contour       # alternates ascending / descending endings
python demo.py --demo progression   # ii-V-I in C, advancing one chord per phrase
python demo.py --demo blues         # 12-bar blues
python demo.py --demo tritone       # V7 always replaced by tritone sub (bII7)
python demo.py --demo pedal         # pedal tone with cycling upper harmony
python demo.py --demo free          # chromatic baseline
```

Options (combine with any demo):

```
--root NOTE    tonic / modal root, e.g. --root Bb   (default: D)
--temp FLOAT   generation temperature                (default: 0.9)
--notes INT    max notes per sax response            (default: 12)
```

The console prints the chord name, scale size, swing bias, detected bass feel, contour, and output velocity for each response.

## Automated test suite

`tests/run_tests.py` runs ten feature tests programmatically — no MIDI hardware needed. Each test constructs a synthetic bass phrase, analyses it, generates a sax response with explicit parameters, and writes a log file. A combined two-track demo MIDI is also produced.

```bash
python tests/run_tests.py
```

Outputs:
- `tests/logs/01_basic_response.txt` … `tests/logs/10_sparse.txt` — per-test log showing bass phrase, analysis features, generation parameters, and sax response
- `tests/demo.mid` — two-track MIDI (bass + sax) with all test segments in sequence, ready to import into a DAW

| Test | Feature exercised |
|------|-------------------|
| 01 | Basic call-response |
| 02 | Dynamics — soft (vel=30) |
| 03 | Dynamics — loud (vel=100) |
| 04 | Contour — ascending bass → descending sax |
| 05 | Contour — descending bass → ascending sax |
| 06 | Swing feel detected → no extra triplet bias |
| 07 | Straight feel detected → triplet-grid bias applied |
| 08 | D Dorian scale pitch bias |
| 09 | ii-V-I: same phrase over Dm7, G7, Cmaj |
| 10 | Sparse phrase → sax generates longer response |

## Project structure

```
wolfson/
├── main.py                       Entry point (full 5-minute performance)
├── demo.py                       Feature-focused testing without the arc
├── test-midi-in.py               Verify MIDI input port and event routing
├── config.py                     All tunable parameters
├── wolfson_train.ipynb           Google Colab training notebook
├── requirements.txt
├── input/
│   ├── midi_listener.py          MIDI input, note events
│   ├── phrase_detector.py        Segments note stream into phrases
│   ├── phrase_analyzer.py        Phrase features: contour, density, Q&A type, swing ratio, dynamics
│   └── beat_estimator.py         Live tempo estimation from bass onsets
├── memory/
│   └── phrase_memory.py          Stores phrases for recall and development
├── generator/
│   ├── lstm_model.py             LSTM with chord conditioning
│   ├── phrase_generator.py       Seeds LSTM; contour, scale, and swing steering
│   └── train.py                  Training script
├── controller/
│   ├── arc_controller.py         Arc, leadership, proactive mode
│   └── harmony.py                Harmonic modes: free, modal, progression, pedal
├── output/
│   └── midi_output.py            Per-note MIDI playback with articulation
├── data/
│   ├── encoding.py               Pitch+duration token encoding
│   ├── chords.py                 Chord parsing and 49-token vocabulary
│   ├── scales.py                 Mode interval tables and scale pitch-class helpers
│   ├── instruments.py            Instrument family definitions and pitch ranges
│   └── prepare.py                WJD data preparation script
└── tests/
    ├── run_tests.py              Automated test suite; writes logs + demo.mid
    └── logs/                     Per-test log files (generated)
```

## Extending to other instruments

```bash
python data/prepare.py --instrument trumpet
python generator/train.py --instrument trumpet
```

Set `DEFAULT_INSTRUMENT = "trumpet"` in `config.py`, or pass `instrument="trumpet"` to `PhraseGenerator`. Families and pitch ranges are in `data/instruments.py`.

## Acknowledgements

Solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/), Jazzomat Research Project, Hochschule für Musik Franz Liszt Weimar.

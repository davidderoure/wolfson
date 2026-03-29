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
                                          PhraseGenerator
                                       (LSTM + chord conditioning
                                        + contour steering)
                                                 │
                                           MidiOutput ──► Synth (sax voice)
```

### Musicality features

**Phrase analysis** — each bass phrase is characterised by note density, pitch ambitus, contour slope, and rhetorical type (question / answer / neutral). The arc controller uses these to shape its response.

**Question and answer** — the sax detects whether a bass phrase is a question (rising, open ending) or an answer (falling, resolving), and responds with the complement. This creates the classic jazz call-and-response dialogue.

**Leadership and role swapping** — the system tracks who is leading at each moment. Sparse bass playing (low density, small range) signals the bassist is comping; the sax takes the initiative. Dense melodic bass signals the sax should respond. Leadership shifts deliberately over the arc.

**Contour steering** — soft logit biases applied to pitch tokens in the final portion of each generated phrase guide its ending upward or downward as needed.

**Chord conditioning** — the LSTM is trained with chord context from the WJD beats table (49-token chord vocabulary: 12 roots × 4 quality classes + NC). At runtime, chord index can be provided from a chart; defaults to NC (no-chord) if absent — the model degrades gracefully.

**Live tempo tracking** — a `BeatEstimator` infers BPM from inter-onset intervals in the bass line using IOI histogram autocorrelation. Generated sax phrases play back in time with the bassist's actual tempo. No click track or advance setup required.

**Proactive mode** — the sax does not always wait for a bass phrase to end. When the bassist is sparse or silent, the sax initiates. During the resolution stage, the sax always plays the final phrase.

### Performance arc

| Time | Stage | Character |
|------|-------|-----------|
| 0:00–1:00 | sparse | Short exchanges; bass leads; establish motifs |
| 1:00–2:30 | building | Longer phrases; sax begins recalling and initiating |
| 2:30–3:30 | peak | Maximum density; sax leads; adventurous generation |
| 3:30–4:30 | recapitulation | Return to early phrases, transformed; role reversal |
| 4:30–5:00 | resolution | Sax thins and resolves; plays the last phrase |

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

## Project structure

```
wolfson/
├── main.py                       Entry point
├── config.py                     All tunable parameters
├── wolfson_train.ipynb           Google Colab training notebook
├── requirements.txt
├── input/
│   ├── midi_listener.py          MIDI input, note events
│   ├── phrase_detector.py        Segments note stream into phrases
│   ├── phrase_analyzer.py        Phrase features: contour, density, Q&A type
│   └── beat_estimator.py         Live tempo estimation from bass onsets
├── memory/
│   └── phrase_memory.py          Stores phrases for recall and development
├── generator/
│   ├── lstm_model.py             LSTM with chord conditioning
│   ├── phrase_generator.py       Seeds LSTM, contour steering, sampling
│   └── train.py                  Training script
├── controller/
│   └── arc_controller.py         Arc, leadership, proactive mode
├── output/
│   └── midi_output.py            Per-note MIDI playback with articulation
└── data/
    ├── encoding.py               Pitch+duration token encoding
    ├── chords.py                 Chord parsing and vocabulary
    ├── instruments.py            Instrument family definitions
    └── prepare.py                WJD data preparation script
```

## Extending to other instruments

```bash
python data/prepare.py --instrument trumpet
python generator/train.py --instrument trumpet
```

Set `DEFAULT_INSTRUMENT = "trumpet"` in `config.py`, or pass `instrument="trumpet"` to `PhraseGenerator`. Families and pitch ranges are in `data/instruments.py`.

## Acknowledgements

Solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/), Jazzomat Research Project, Hochschule für Musik Franz Liszt Weimar.

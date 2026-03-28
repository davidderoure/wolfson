# Wolfson

An interactive jazz improvisation system. You play bass; it plays sax. It listens, responds, and develops a musical conversation with you over a 5-minute performance arc.

## Overview

Wolfson uses an LSTM trained on jazz solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) to generate melodic sax responses to live bass input. The system is designed for live performance: it detects phrases in your playing, generates a response, and manages a structural arc (sparse → building → peak → recapitulation → resolution) over the duration of the piece.

### Architecture

```
Bass (pitch-to-MIDI) → MidiListener → PhraseDetector
                                           │
                                     PhraseMemory ← stores both voices
                                           │
                                    ArcController   ← tracks elapsed time,
                                           │           decides response mode
                                    PhraseGenerator ← LSTM model
                                           │
                                      MidiOutput → Synth (sax voice)
```

The LSTM operates at the phrase level: it takes the bass phrase as a seed and generates a melodic sax response. Phrases are encoded as interleaved pitch + duration token sequences, with duration represented beat-relative (tempo-independent) using log-scale buckets — preserving expressive timing from the training corpus.

The `ArcController` manages the macro structure:

| Time | Stage | Character |
|------|-------|-----------|
| 0:00–1:00 | sparse | Short exchanges, establish motifs |
| 1:00–2:30 | building | Longer phrases, begin recalling earlier material |
| 2:30–3:30 | peak | Maximum density, adventurous generation |
| 3:30–4:30 | recapitulation | Return to early phrases, transformed |
| 4:30–5:00 | resolution | Sparse again, echo the opening |

## Training Data

Solos are sourced from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) (WJD), which contains 456 annotated jazz solo transcriptions. The saxophone subset (alto, tenor, baritone, soprano) comprises 271 solos — ~106,000 notes across ~5,000 phrases.

The system is designed to support multiple instrument families (trumpet, trombone, flute) with separate models for each.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `python-rtmidi`, `torch`, `numpy`, `pretty_midi`

### Hardware

- Monophonic pitch-to-MIDI converter on your instrument (bass, or any monophonic source)
- MIDI interface
- Synth or software instrument on a MIDI output channel for the sax voice

### MIDI configuration

Edit `config.py` to set your MIDI port indices:

```python
MIDI_INPUT_PORT = 0    # your pitch-to-MIDI interface
MIDI_OUTPUT_PORT = 0   # your sax synth
```

Run `python -c "import rtmidi; m=rtmidi.MidiIn(); print(m.get_ports())"` to list available ports.

## Training

### 1. Get the data

Download the Weimar Jazz Database from [jazzomat.hfm-weimar.de](https://jazzomat.hfm-weimar.de/download/download.html):

- `wjazzd.db` — SQLite database (solo metadata and note data)
- `RELEASE2.0_mid_unquant.zip` — unquantised MIDI files (optional, for MIDI-only mode)

Place `wjazzd.db` in `data/raw/`. Extract the zip to `data/raw/midi_unquant/` if using MIDI-only mode.

### 2. Prepare training data

```bash
# Inspect the database (check instrument distribution, schema)
python data/prepare.py --inspect

# Extract saxophone phrases
python data/prepare.py --instrument sax
```

### 3. Train

```bash
python generator/train.py --instrument sax --epochs 100
```

The best model is saved to `models/sax_best.pt`.

**Training on Google Colab:** open `wolfson_train.ipynb` — it handles data upload, dependency installation, training, and model download.

### 4. Run

```bash
python main.py
```

## Project structure

```
wolfson/
├── main.py                    Entry point
├── config.py                  All tunable parameters
├── wolfson_train.ipynb        Google Colab training notebook
├── requirements.txt
├── input/
│   ├── midi_listener.py       MIDI input, note events
│   └── phrase_detector.py     Segments note stream into phrases
├── memory/
│   └── phrase_memory.py       Stores phrases for recall and development
├── generator/
│   ├── lstm_model.py          LSTM model definition
│   ├── phrase_generator.py    Seeds LSTM, samples output
│   └── train.py               Training script
├── controller/
│   └── arc_controller.py      5-minute structural arc
├── output/
│   └── midi_output.py         MIDI output
└── data/
    ├── encoding.py             Pitch+duration token encoding
    ├── instruments.py          Instrument family definitions
    └── prepare.py              WJD data preparation script
```

## Extending to other instruments

To train a trumpet model, for example:

```bash
python data/prepare.py --instrument trumpet
python generator/train.py --instrument trumpet
```

Then pass `instrument="trumpet"` when constructing `PhraseGenerator` in `main.py`. Instrument families and pitch ranges are defined in `data/instruments.py`.

## Acknowledgements

Solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/), Jazzomat Research Project, Hochschule für Musik Franz Liszt Weimar.

# Wolfson

An interactive jazz improvisation system. You play bass; it plays sax. It listens, responds, and develops a musical conversation with you over a 5-minute performance arc.

## Overview

Wolfson uses an LSTM trained on jazz solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) to generate melodic sax responses to live bass input. The system is designed for live performance: it detects phrases in your playing, generates a response, and manages a structural arc over the duration of the piece.

### Architecture

```
Bass (pitch-to-MIDI) ──► MidiListener ──► PhraseDetector ──► PhraseAnalyzer
                               │                               (contour, density,
                          BeatEstimator                         Q&A type, swing,
                         (live tempo)                           dynamics, energy
                               │                                profile, pitch
                               │                                classes, interval
                               │                                motifs, lyrical
                               │                                motifs)
                               │                                     │
                               │                               PhraseMemory
                               │                            (phrases + motifs
                               │                             + lyrical motifs,
                               │                             both voices)
                               │                                     │
                               └──────────── ArcController ──────────┘
                                          (5-min arc, leadership,
                                           proactive mode,
                                           bass pitch-class tracking,
                                           energy arc selection,
                                           motif + lyrical motif selection,
                                           stage swing baseline)
                                                 │
                                        HarmonyController
                                    (mode, progression, pedal,
                                     tritone substitution)
                                                 │
                                          PhraseGenerator
                                    (LSTM + chord conditioning
                                     + pitch range limits
                                     + register gravity
                                     + scale pitch bias
                                     + contour steering
                                     + swing/triplet bias
                                     + energy arc shaping
                                     + long-note penalty
                                     + singable duration bias
                                     + motivic development
                                     + voice leading
                                     + modal leap bonus (P4/P5)
                                     + rest injection
                                     + beat accumulator)
                                                 │
                               ┌─────────────────┴──────────────────┐
                          MidiOutput                            OscOutput
                     (per-note velocity                   (phrase events to
                      from energy arc)                  TouchDesigner / Max
                          │                              / Processing etc.)
                          ▼
                    Synth (sax voice)
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

**Swing / triplet feel** — the system detects whether the bass is playing straight or swung (from consecutive IOI ratios) and produces a reactive swing bias (1.0 for straight, 0.0 for swung, 0.3 for mixed). This reactive value is then clamped to a per-stage (min, max) band rather than used directly. The band enforces the arc's swing character regardless of what the detector reads: sparse stays exploratory and barely swung (ceiling 0.25) even if the detected feel is "straight"; resolution stays a gentle ballad lilt (0.10–0.30); building warms up through a groove range (0.30–0.65); peak is always hard bebop swing (0.70). This also breaks the self-play oscillation where lyrical long notes look "straight" to the detector and previously pushed the swing to 1.0 at every stage.

**Live tempo tracking** — a `BeatEstimator` infers BPM from inter-onset intervals in the bass line. Generated sax phrases play back in time with the bassist's actual tempo. No click track or advance setup required.

**Dynamics** — the mean MIDI velocity of each bass phrase is tracked and mapped to the sax output velocity, so the sax mirrors the bassist's dynamic level. A stage multiplier modulates this further: sparse and resolution stages are inherently softer; the peak stage pushes louder. Playing quietly draws a quiet response; playing hard drives the sax to match.

**Articulation** — each generated note sounds for 85% of its time slot, with a short silence before the next note. A minimum duration floor (0.2 beats, ≈100 ms at 120 BPM) prevents imperceptibly short notes from appearing in dense generated passages.

**Bass pitch-class tracking** — the system extracts the pitch-class set of each bass phrase (which of the 12 chromatic pitch classes appeared) and uses it to steer the sax toward the tonality the bassist is actually implying. With ≥ 4 distinct pitch classes (a scale fragment or lick), the bass pitch classes override the arc's harmonic plan entirely. With 2–3 (a motif or interval), they are unioned with the arc scale to broaden the available palette. With fewer, the arc's harmonic plan is used unchanged. This means that switching from D Dorian to Gb pentatonic produces an immediate shift in the sax response. The scale source (`bass` / `blend` / `arc`) is shown in the console and dashboard on every phrase.

**Energy arc** — every generated phrase has an internal energy shape applied by position-dependent logit biases. The bass phrase's own energy profile (classified as `arch`, `ramp_up`, `ramp_down`, `spike`, or `flat` from its per-note pitch and velocity trajectory) is complemented: a bass `ramp_up` gets a sax `arch` (peaks then resolves); a `ramp_down` bass gets a `ramp_up` sax. Stage overrides apply at structural boundaries: peak stage always forces `arch`; resolution always forces `ramp_down`. The arc shapes three things simultaneously — pitch register (higher = more energetic), note density (shorter durations at the peak), and per-note velocity (0.75×–1.25× the base phrase velocity). The current arc shape is shown in the console and dashboard on every phrase.

**Motivic development** — the system tracks all 2-, 3-, and 4-note interval patterns (n-grams) across recent phrases in transposition-invariant form (signed semitones between adjacent pitches, not absolute pitch). When a pattern has appeared at least twice in the last 16 phrases, it is passed to the generator as a `motif_target`. A logit bias fires whenever the generated line has entered a prefix of the pattern — nudging the LSTM to complete the interval sequence. Strength scales with stage: zero in the sparse stage (insufficient material), rising through building and peak, strongest in recapitulation (0.8) where thematic return is most meaningful. This creates audible motivic echoes and development across the full arc without forcing the model.

**Lyrical motif re-use** — a second, narrower motif bank stores interval patterns extracted *only* from sustained (singable) notes: sax notes whose `duration_beats` is at or above 0.4 beats. During quieter arc stages (recapitulation at strength 0.7, resolution at 0.5) these lyrical motifs are given priority over the general pool, so the sax preferentially quotes back the melodic shapes from its own long-note passages. The effect is a sense of returning home to the singable themes built earlier in the arc, rather than recycling ornamental fast-note fragments.

**Voice leading** — three complementary biases operate on pitch tokens at every step. Chord-tone targeting adds a positive bias to root, 3rd, and 7th pitch classes, growing linearly with `arc_position` from 0 at the phrase start to full strength at the end — so the model is free in the phrase body but is nudged toward harmonic resolution at the cadence. Stepwise motion preference adds a small constant bias toward pitches ±1–2 semitones from the last generated pitch. Leap incentive adds a small positive bonus for 3rd–5th intervals (3–7 semitones), increasing melodic shape and reducing chromatic saturation. All three are calibrated within a shared logit budget so they complement rather than override the LSTM's learned jazz vocabulary.

**Modal leap bonus** — in modal stages (building and peak), an additional logit boost is applied to P4 (perfect 4th, 5 semitones) and P5 (perfect 5th, 7 semitones) intervals, scaled by a `modal_strength` value set per stage (0.0 at sparse/resolution, 0.6 at building, 1.0 at peak, 0.4 at recapitulation). This reflects the quartal and pentatonic character of modal jazz — the stacked-4ths vocabulary of McCoy Tyner, Herbie Hancock, and Wayne Shorter — which is under-represented in the base LSTM's learned distribution but clearly differentiates modal from tonal playing. Statistical analysis of output MIDI confirmed P4 motion rises from ~7% of intervals at modal_strength=0 to ~21% at modal_strength=1.0, matching the expected interval profile for modal jazz.

**Phrase breathing** — after the LSTM sampling loop, silence sentinels (REST_PITCH = −1) are spliced into the generated phrase with a bell-curve probability distribution: zero at the very start and end of the phrase, peaking at the midpoint (max 15% chance per inter-note gap). Rest duration is drawn from {0.5, 1.0} beats. The MidiOutput layer translates sentinels to `time.sleep()` calls with no MIDI note sent. This models the breathing and phrasing pauses a human saxophonist would naturally insert — continuous eighth-note streams without gaps are musically inauthentic regardless of harmonic accuracy.

**Singable duration bias** — the LSTM's training data is dominated by 8th and 16th notes, giving it a strong prior toward fast, busy output. To counter this, a bell-curve logit boost is applied to the duration token vocabulary, centred on the quarter-note token (≈ 0.95 beats) with a width of ±4.5 tokens, pulling sampling toward the 0.5–1.7 beat range where sustained, melodic, "singable" lines live. The boost is scaled by `(1 − rhythmic_density)`, so it is strongest in lyrical stages (full strength at resolution, density=0.1) and progressively suppressed toward the busiest stage (density=0.9 at peak). The result is that mid-register quarter-note lines dominate at sparse and resolution stages while the peak can still drive faster runs.

**Rhythmic density** — the arc controller emits a `rhythmic_density` value (0.0 = lyrical, 1.0 = bebop) for each stage, which the generator uses to scale the singable-duration bias inversely. This creates a natural busyness gradient across the arc: sparse and resolution feel spacious and melodic; peak is the most rhythmically active. The value is logged in the console output alongside `modal_strength`.

| Stage | Rhythmic density | Character |
|-------|-----------------|-----------|
| sparse | 0.2 | Open, sustained — establish motifs slowly |
| building | 0.5 | Mixed — lines begin to flow |
| peak | 0.9 | Fast and active — maximum rhythmic intensity |
| recapitulation | 0.3 | Lyrical return — recall themes with space |
| resolution | 0.1 | Slowest — long tones, fading out |

**Phrase length control** — four cooperating mechanisms manage phrase length. A duration-token penalty tensor applies a graduated negative logit bias to tokens above ~2 beats. A bell-curve singable-duration boost raises the floor toward quarter notes (see above). A beat accumulator hard-stops generation once the total accumulated beats reaches `max_phrase_beats` (default `MAX_PHRASE_BEATS = 16`). The arc controller caps `n_notes` at 14, uses stage-scaled multipliers (1.1× / 0.75× for leading/following), and enforces a per-stage minimum phrase length floor (4–8 notes) so even the shortest "following" phrase has enough notes to be musically complete.

**Beat-matching / trading bars** — pass `--trade` to enable beat-matching mode. The sax response is capped to the same number of beats as the incoming bass phrase, so trading 2s, 4s, or 8s emerges naturally from however long the bassist chooses to play. A bassist who plays a tight 2-bar phrase (8 beats at 120 BPM) gets an 8-beat sax response; a long 8-bar phrase gets a matching 8-bar response. A minimum floor (`TRADE_BEATS_MIN = 2.0`) prevents very short fills from producing unmusically brief responses. The beat cap is computed from the detected tempo, so it tracks tempo changes automatically. The console log shows `cap=Nb` when trade mode is active. In self-play mode `--trade` creates stable back-and-forth exchanges of equal length.

**Repetition control** — a growing logit penalty is applied to the most recently played pitch token, starting at −2.5 logits on the first immediate repeat and adding −2.0 for each further consecutive repeat. After three same-pitch notes in a row the penalty reaches −6.5 logits, effectively eliminating a fourth repeat while still permitting occasional passing-tone ornaments. The stepwise bias strength was simultaneously reduced (0.4 → 0.1) to remove a partial cancellation that was blunting the penalty's effect.

**Proactive mode** — the sax does not always wait for a bass phrase to end. When the bassist is sparse or silent, the sax initiates. During the resolution stage, the sax always plays the final phrase.

### Performance arc

| Time | Stage | Harmonic mode | Modal strength | Rhythmic density | Swing range | Character |
|------|-------|---------------|----------------|-----------------|-------------|-----------|
| 0:00–1:00 | sparse | free | 0.0 | 0.2 | 0.00–0.25 | Exploratory, barely swung; establish motifs slowly |
| 1:00–2:30 | building | modal | 0.6 | 0.5 | 0.30–0.65 | Groove warming up; P4/P5 leaps rise |
| 2:30–3:30 | peak | progression | 1.0 | 0.9 | 0.70 | Hard bebop swing; maximum activity; tritone subs |
| 3:30–4:30 | recapitulation | modal | 0.4 | 0.3 | 0.30–0.60 | Medium swing; lyrical return; singable motifs recalled |
| 4:30–5:00 | resolution | pedal | 0.0 | 0.1 | 0.10–0.30 | Gentle ballad lilt; slow, sustained; sax plays the final phrase |

## Training data

Solos from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) (WJD). The saxophone subset (alto, tenor, baritone, soprano) contains 271 solos — ~106,000 notes across ~5,000 phrases. Chord changes are extracted per note from the WJD beats table.

The system supports multiple instrument families (trumpet, trombone, flute) with separate models.

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Dependencies: `python-rtmidi`, `torch`, `numpy`, `pretty_midi`, `rich`, `python-osc`

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
python main.py --trade          # beat-matching: sax matches bass phrase length
```

#### Dashboard

Pass `--dashboard` for a full-screen rich terminal display showing the arc progress bar, current stage, harmonic mode, scale tracking, last-phrase notes, and rolling statistics — suitable for a personal monitor while performing. The dashboard uses a forced black background with high-contrast white and cyan text so it remains legible in all lighting conditions.

```bash
python main.py --dashboard
python main.py --self-play --dashboard
```

#### Audience web display

Pass `--web` to serve a mobile-optimised display page on the local network. Any audience member on the same WiFi can open the URL in their phone browser — no app required.

```bash
python main.py --web                      # serves on port 5000
python main.py --web --web-port 8080      # custom port (macOS: port 5000 may be taken by AirPlay)
python main.py --self-play --web --dashboard
```

At startup the terminal prints the URL to share:

```
Audience display (local):   http://192.168.1.42:5000
  Share with audience on the same WiFi network.
```

The page shows:
- **Arc progress bar** — five colour-coded segments (sparse grey, building cyan, peak red, recapitulation green, resolution yellow); each segment fills as the performance progresses with the current stage partially highlighted
- **Stage name** in large type, coloured by stage, with time remaining
- **BPM** and phrase count as large numbers
- **Harmony, scale, contour, velocity** as info cards
- **Last phrase note names** as coloured chips (in the current stage colour)
- **Trigger indicator** — `⟵ bass phrase` or `◎ sax initiates`
- **Pulse animation** — a brief coloured border flash on every new phrase
- Auto-reconnects silently if the connection drops

The page is fully self-contained (no CDN dependencies) and loads instantly on a slow venue connection. Updates are pushed via Server-Sent Events — the page is live without polling.

##### Public tunnel (eduroam / institutional networks)

On eduroam or other networks where you cannot set up a local AP, use `--tunnel` to open a [cloudflared quick tunnel](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/do-more-with-tunnels/trycloudflare/). This makes an outbound HTTPS connection to Cloudflare's edge — no inbound ports required, no account needed.

**Install cloudflared first:**
```bash
brew install cloudflare/cloudflare/cloudflared
```

**Run with tunnel:**
```bash
python main.py --web --tunnel
python main.py --web --web-port 8080 --tunnel
```

At startup both URLs are printed:
```
Audience display (local):   http://192.168.1.42:8080
  Share with audience on the same WiFi network.
Audience display (public):  https://curious-fox-amazing.trycloudflare.com
  Share this URL with the audience anywhere.
```

The public URL changes each run (Cloudflare assigns a random subdomain). Share it via a short URL, QR code, or just read it out. All traffic is end-to-end HTTPS via Cloudflare — the Flask server itself stays on localhost.

#### OSC output

Pass `--osc-host` to broadcast phrase events over UDP/OSC to any receiver — TouchDesigner, Max/MSP, Processing, Ableton Live, etc. Every phrase sends a burst of messages covering arc position, tempo, harmony, scale source, contour, velocity, and pitch list. See `output/osc_output.py` for the full address scheme.

```bash
python main.py --osc-host 127.0.0.1              # localhost (default port 9000)
python main.py --osc-host 192.168.1.10           # separate machine on LAN
python main.py --osc-host 127.0.0.1 --osc-port 8000
```

Key OSC addresses:

| Address | Type | Description |
|---------|------|-------------|
| `/wolfson/arc/stage` | s | Current stage name |
| `/wolfson/arc/progress` | f | 0.0–1.0 through the full arc |
| `/wolfson/bpm` | f | Live tempo estimate |
| `/wolfson/harm/mode` | s | `free` / `modal` / `progression` / `pedal` |
| `/wolfson/scale/source` | s | `bass` / `blend` / `arc` |
| `/wolfson/scale/source_f` | f | 0.0=arc  0.5=blend  1.0=bass |
| `/wolfson/phrase/contour_f` | f | −1.0=desc  0.0=neutral  1.0=asc |
| `/wolfson/phrase/velocity` | i | MIDI velocity (40–110) |
| `/wolfson/pitches` | i… | One int per note in the phrase |

#### Self-play mode

Pass `--self-play` to run Wolfson autonomously — no MIDI input hardware needed. The sax feeds its own output back as input, creating a continuous generative loop. The 5-minute structural arc still governs the performance.

```bash
python main.py --self-play           # 120 BPM
python main.py --self-play --bpm 90  # set tempo
```

The system seeds itself with a short D minor pentatonic motif, then responds to each phrase it generates. Each response seeds the next, so the musical conversation develops and evolves over the full arc. Useful for:
- Listening to the model's musical character without a musician present
- Testing the arc structure and harmonic progression in real time
- Leaving it running as a generative ambient piece

**Two-channel dialogue** — in self-play mode, phrases alternate between MIDI channel 1 and MIDI channel 2 (configurable as `SELF_PLAY_CH_A` / `SELF_PLAY_CH_B` in `config.py`). Route channel 1 to one synth voice (e.g. alto sax) and channel 2 to another (e.g. tenor sax, or a contrasting timbre) to make the call-and-response structure directly audible. Odd-numbered phrases play on channel 1; even-numbered phrases play on channel 2. In live-bass mode the sax always plays on channel 1.

Console output is identical to normal mode: each phrase logs stage, tempo, MIDI channel, leadership, harmonic mode, scale source, contour, velocity, modal strength, rhythmic density, and note count. A rolling statistics block is printed every 8 phrases. Use `--dashboard` for the full-screen display.

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
- `tests/logs/01_basic_response.txt` … `tests/logs/21_voice_leading_G7.txt` — per-test log showing bass phrase, analysis features, generation parameters, and sax response (including per-note velocity scale and detected energy profile)
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
| 11 | Tempo — 60 BPM (slow) |
| 12 | Tempo — 90 BPM |
| 13 | Tempo — 160 BPM |
| 14 | Tempo — 200 BPM (fast) |
| 15 | Energy arc — ramp_up bass → arch sax (auto-detected + complemented) |
| 16 | Energy arc — arch (forced); per-note velocity peaks at midpoint |
| 17 | Energy arc — ramp_down (forced); velocity and register descend |
| 18 | Motivic development — ascending minor third motif (3, 2) at strength 0.8 |
| 19 | Motivic development — blues cell motif (3, -2, -1) at strength 0.8 |
| 20 | Voice leading — Dm7 endpoint targeting (chord tones D, F, C) |
| 21 | Voice leading — G7 endpoint targeting (chord tones G, B, F) |

## Project structure

```
wolfson/
├── main.py                       Entry point (full 5-minute performance)
├── demo.py                       Feature-focused testing without the arc
├── test-midi-in.py               Verify MIDI input port and event routing
├── osc-monitor.py                Print incoming OSC messages (test OSC output)
├── config.py                     All tunable parameters (incl. SELF_PLAY_CH_A/B, TRADE_BEATS_MIN)
├── wolfson_train.ipynb           Google Colab training notebook
├── requirements.txt
├── input/
│   ├── midi_listener.py          MIDI input, note events
│   ├── phrase_detector.py        Segments note stream into phrases
│   ├── phrase_analyzer.py        Phrase features: contour, density, Q&A type, swing, dynamics, energy profile, interval motifs, lyrical motifs
│   └── beat_estimator.py         Live tempo estimation from bass onsets
├── memory/
│   └── phrase_memory.py          Stores phrases + motifs + lyrical motifs for recall and development
├── generator/
│   ├── lstm_model.py             LSTM with chord conditioning
│   ├── phrase_generator.py       Seeds LSTM; contour, scale, swing, energy arc, motif, voice leading, modal leap, singable duration bias, rest injection, phrase length control
│   └── train.py                  Training script
├── controller/
│   ├── arc_controller.py         Arc, leadership, proactive mode, modal_strength + rhythmic_density + swing range schedules, lyrical motif recall
│   └── harmony.py                Harmonic modes: free, modal, progression, pedal
├── output/
│   ├── midi_output.py            Per-note MIDI playback with articulation
│   ├── dashboard.py              Rich full-screen terminal display, black background, high-contrast (--dashboard)
│   ├── web_display.py            Audience web display served over HTTP/SSE (--web)
│   └── osc_output.py             UDP/OSC phrase events for stage visuals (--osc-host)
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

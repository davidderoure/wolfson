# Wolfson

An interactive jazz improvisation system. You play bass; it plays sax. It listens, responds, and develops a musical conversation with you over a 5-minute performance arc.

## Overview

Wolfson uses an LSTM trained on jazz solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/) to generate melodic sax responses to live bass input. The system is designed for live performance: it detects phrases in your playing, generates a response, and manages a structural arc over the duration of the piece.

### Architecture

```
Bass (pitch-to-MIDI) ──► MidiListener ──────────► PhraseDetector ──► PhraseAnalyzer
                         (pitch range,             (note dur min,     (contour, density,
                          velocity min)             min phrase notes,  Q&A type, swing,
                               │                   monophony,         dynamics, energy
                          BeatEstimator             watchdog)          profile, pitch
                         (live tempo;                                  classes, interval
                          last_onset_time                              motifs, lyrical
                          for beat-sync)                               motifs)
                               │   every note_on:                              │
                               │   arc.touch_bass()                            │
                               │   (keeps time_since_bass               PhraseMemory
                               │    current mid-phrase)              (phrases + motifs
                               │                                     + lyrical motifs,
                               │                                      both voices)
                               │                                             │
                               └──────────────── ArcController ──────────────┘
          proactive loop ─────►              (5-min arc, leadership,
       (0.5 s; fires only when               proactive mode,
        bass paused; live phrase             bass pitch-class tracking,
        peeking; beat-sync to                energy arc selection,
        next beat boundary)                 motif + lyrical motif selection,
                                             rhythmic density + complementarity,
                                             register contrast scheduling,
                                             stage swing baseline,
                                             opening echo: sparse stage sets
                                              echo_phrase, bypasses generator)
                                                       │
                                              HarmonyController
                                          (mode, progression, pedal,
                                           tritone substitution)
                                                       │
                                                PhraseGenerator
                                          (LSTM + chord conditioning
                                           + pitch range limits
                                           + register gravity
                                           + register contrast
                                           + scale pitch bias
                                           + contour steering
                                           + swing/triplet bias
                                           + energy arc shaping
                                           + long-note penalty
                                           + singable duration bias
                                           + motivic development
                                           + voice leading
                                           + modal leap bonus (P4/P5)
                                           + repetition penalty
                                           + rest injection
                                           + beat accumulator
                                           + rhythmic displacement)
                                                       │ full phrase
                     ┌─────────────────────────────────┼──────────────────────────┐
                     │                                 │                          │
               PhraseMemory              WebAudienceDisplay /             performance
             (motifs, recall,            OscOutput / Dashboard              thinning
              self-play seed)           (always see the full            (short notes dropped
                                          intended phrase)               stochastically at
                                                                          output only;
                                                                        +velocity jitter on
                                                                         sax riff replays)
                                                                               │
                                                                          MidiOutput
                                                                      (queue architecture:
                                                                       single output thread;
                                                                       "latest wins" — new
                                                                       phrase supersedes
                                                                       previous mid-note;
                                                                       on_complete callback
                                                                       defers arc timing;
                                                                       per-note velocity:
                                                                       energy arc ×
                                                                       peak accent ×
                                                                       end taper)
                                                                      ┌─────┴──────┐
                                                                      │            │
                                                                 sax voice    chord hint
                                                                 (ch 1/2)     (ch 3,
                                                                              --chord-hint)
```

### Musicality features

**Phrase analysis** — each bass phrase is characterised by note density, pitch ambitus, contour slope, and rhetorical type (question / answer / neutral). The arc controller uses these to shape its response.

**Question and answer** — the sax detects whether a bass phrase is a question (rising, open ending) or an answer (falling, resolving), and responds with the complement. This creates the classic jazz call-and-response dialogue.

**Leadership and role swapping** — the system tracks who is leading at each moment. Sparse bass playing (low density, small range) signals the bassist is comping; the sax takes the initiative. Dense melodic bass signals the sax should respond. Leadership shifts deliberately over the arc.

**Opening echo** — for the first few bass phrases during the sparse opening stage, Wolfson plays the bass phrase back literally rather than generating a new response. The phrase is transposed to the sax register by the nearest whole-octave shift, so every interval and rhythm is preserved exactly and the audience hears the same melodic idea in the saxophone's voice. This makes the call-and-response relationship immediately legible even to listeners unfamiliar with free jazz — the echo is the most direct possible demonstration that the system is listening. After `ECHO_MAX_EXCHANGES` (default: 4) echoes the budget expires and Wolfson begins generating its own material; because the echoed phrases were stored in memory like any other response, early motivic development draws directly from the bass's opening ideas. `ECHO_PROBABILITY` (default: 0.85 in performance, 1.0 for testing) controls the per-phrase chance of echoing — a value below 1.0 means the system occasionally responds independently from the first exchange, preventing the echo from feeling mechanical.

**Contour steering** — soft logit biases on pitch tokens guide the sax phrase toward a target contour (ascending / descending / neutral). Steering is applied from the first generated note using the seed phrase's mean pitch as the reference, so the whole arc of the response reflects the rhetorical intent rather than just the tail. Bias strength increases linearly with distance from the reference pitch.

**Chord conditioning** — the LSTM is trained with chord context from the WJD beats table (49-token chord vocabulary: 12 roots × 4 quality classes + NC). At runtime the `HarmonyController` issues a chord index each phrase; the model degrades gracefully to NC if no chord is supplied.

**Harmonic modes** — the `HarmonyController` operates in one of four modes, selected automatically by the performance stage:

| Mode | Behaviour |
|------|-----------|
| `free` | No harmonic steering; chromatic scale bias (sparse stage) |
| `modal` | Root + mode (e.g. D Dorian) held for N phrases, then drifts by a semitone; used in building and recapitulation |
| `progression` | Steps through a chord progression one chord per phrase — ii-V-I, VI-II-V-I, I-VI-II-V, or 12-bar blues; tritone substitution on V7 chords (~35%); used at peak |
| `pedal` | Fixed bass pedal tone with cycling upper harmony (i → bVII7 → i → V7); used in resolution |

**Scale pitch bias** — positive logit bias is added to pitch tokens whose pitch class belongs to the current chord's scale or mode. Non-scale tones are not penalised, so chromatic passing notes remain available. The bias is cadence-shaped rather than flat: 0.5 logits during the first 60% of a phrase (giving the LSTM room to complete its own chromatic patterns from training), then ramping linearly to 3.0 logits by the final note. This means the resolution back to scale tones is stronger and more decisive than a flat bias would produce, while chromatic departures in the phrase body are less likely to be cut short before they can resolve naturally.

**Pitch range** — a soft logit penalty steers generated pitches into a practical sax register (E3–E6). Notes outside this range are penalised proportionally to their distance from the limit, preventing the generator from following the bass into an unplayable register while still allowing occasional extremes.

**Swing / triplet feel** — the system detects whether the bass is playing straight or swung (from consecutive IOI ratios) and produces a reactive swing bias (1.0 for straight, 0.0 for swung, 0.3 for mixed). This reactive value is then clamped to a per-stage (min, max) band rather than used directly. The band enforces the arc's swing character regardless of what the detector reads: sparse stays exploratory and barely swung (ceiling 0.25) even if the detected feel is "straight"; resolution stays a gentle ballad lilt (0.10–0.30); building warms up through a groove range (0.30–0.65); peak is always hard bebop swing (0.70). This also breaks the self-play oscillation where lyrical long notes look "straight" to the detector and previously pushed the swing to 1.0 at every stage.

**Live tempo tracking** — a `BeatEstimator` infers BPM from inter-onset intervals in the bass line. Generated sax phrases play back in time with the bassist's actual tempo. No click track or advance setup required. The algorithm is currently tuned for tempos of **55 BPM and above**; below this threshold note-timing calculations break down and the output becomes unreliable.

**Dynamics** — the mean MIDI velocity of each bass phrase is tracked and mapped to the sax output velocity, so the sax mirrors the bassist's dynamic level. A stage multiplier modulates this further: sparse and resolution stages are inherently softer; the peak stage pushes louder. Playing quietly draws a quiet response; playing hard drives the sax to match.

**Generation temperature** — the LSTM samples each note token from a probability distribution; temperature controls how peaked or flat that distribution is. The arc varies temperature by stage (0.70 at resolution → 1.05 at peak) to match the structural character of each section. The `--temperature` flag adds an offset to all stage values, shifting the whole curve up or down while preserving the arc's relative shape. Positive values (e.g. `--temperature 0.2`) produce more adventurous, unpredictable lines with wider interval leaps; negative values (e.g. `--temperature -0.2`) produce more idiomatic, conservative lines that stay closer to the most probable jazz vocabulary. Practical range is roughly ±0.3; the value is clamped to a minimum of 0.1 to prevent degenerate greedy decoding. Empirical self-play tests reveal a tempo-dependent sensitivity: at **60 BPM** the default temperatures sit in a qualitatively distinct lyrical regime — any offset (even ±0.1) collapses the output into a busier, denser character, so the recommended setting at 60 BPM is `--temperature 0.0`. At **90 BPM** the output is less sensitive; `--temperature -0.1` is within natural run-to-run variation overall but produces a slightly more lyrical recapitulation stage. Offsets of −0.2 or below are counterproductive at 90 BPM: lower temperature concentrates sampling on the most frequent training tokens which, at a faster tempo, correspond to shorter absolute note durations, paradoxically making the output busier rather than more conservative.

**Phrase-shape dynamics** — after generation, two post-processing passes give each phrase a natural dynamic arc. First, the highest-pitch note receives a ×1.15 velocity boost (melodic peak accent), reflecting the jazz tendency to push dynamically at the top of a line. Second, the last two pitched notes are tapered to 85% and 70% of their computed velocity (phrase-end taper), simulating breath pressure releasing and giving each phrase a sense of completion rather than an abrupt cutoff. Both passes layer on top of the energy-arc per-note velocity_scale, so they augment rather than override the existing phrase shape.

**Articulation** — each generated note sounds for 85% of its time slot, with a short silence before the next note. A minimum duration floor (0.2 beats, ≈100 ms at 120 BPM) prevents imperceptibly short notes from appearing in dense generated passages.

**Bass pitch-class tracking** — the system extracts the pitch-class set of each bass phrase (which of the 12 chromatic pitch classes appeared) and uses it to steer the sax toward the tonality the bassist is actually implying. With ≥ 4 distinct pitch classes (a scale fragment or lick), the bass pitch classes override the arc's harmonic plan entirely. With 2–3 (a motif or interval), they are unioned with the arc scale to broaden the available palette. With fewer, the arc's harmonic plan is used unchanged. This means that switching from D Dorian to Gb pentatonic produces an immediate shift in the sax response. The scale source (`bass` / `blend` / `arc`) is shown in the console and dashboard on every phrase.

**Energy arc** — every generated phrase has an internal energy shape applied by position-dependent logit biases. The bass phrase's own energy profile (classified as `arch`, `ramp_up`, `ramp_down`, `spike`, or `flat` from its per-note pitch and velocity trajectory) is complemented: a bass `ramp_up` gets a sax `arch` (peaks then resolves); a `ramp_down` bass gets a `ramp_up` sax. Stage overrides apply at structural boundaries: peak stage always forces `arch`; resolution always forces `ramp_down`. The arc shapes three things simultaneously — pitch register (higher = more energetic), note density (shorter durations at the peak), and per-note velocity (0.75×–1.25× the base phrase velocity). The current arc shape is shown in the console and dashboard on every phrase.

**Motivic development** — the system tracks all 2-, 3-, and 4-note interval patterns (n-grams) across recent phrases in transposition-invariant form (signed semitones between adjacent pitches, not absolute pitch). When a pattern has appeared at least twice in the last 16 phrases, it is passed to the generator as a `motif_target`. A logit bias fires whenever the generated line has entered a prefix of the pattern — nudging the LSTM to complete the interval sequence. Strength scales with stage: zero in the sparse stage (insufficient material), rising through building and peak, strongest in recapitulation (0.8) where thematic return is most meaningful. This creates audible motivic echoes and development across the full arc without forcing the model.

**Rhythmic displacement** — pass `--motif-displacement` to randomly shift the motif injection point by 0.5 or 1.0 beats each phrase. The bias is suppressed until the accumulated beat count reaches the displacement offset, so the motif figure can only start after that point. The same interval pattern heard starting on a different beat creates rhythmic variety without changing the melodic content — an eighth-note or quarter-note displacement is enough to place the figure on the offbeat rather than the downbeat. The offset is chosen independently each phrase (random choice of 0.5 or 1.0 beats) so it doesn't become predictable. Off by default.

**Lyrical motif re-use** — a second, narrower motif bank stores interval patterns extracted *only* from sustained (singable) notes: sax notes whose `duration_beats` is at or above 0.4 beats. During quieter arc stages (recapitulation at strength 0.7, resolution at 0.5) these lyrical motifs are given priority over the general pool, so the sax preferentially quotes back the melodic shapes from its own long-note passages. The effect is a sense of returning home to the singable themes built earlier in the arc, rather than recycling ornamental fast-note fragments.

**Voice leading** — three complementary biases operate on pitch tokens at every step. Chord-tone targeting adds a positive bias to root, 3rd, and 7th pitch classes, growing linearly with `arc_position` from 0 at the phrase start to full strength at the end — so the model is free in the phrase body but is nudged toward harmonic resolution at the cadence. Stepwise motion preference adds a small constant bias toward pitches ±1–2 semitones from the last generated pitch. Leap incentive adds a small positive bonus for 3rd–5th intervals (3–7 semitones), increasing melodic shape and reducing chromatic saturation. All three are calibrated within a shared logit budget so they complement rather than override the LSTM's learned jazz vocabulary.

**Modal leap bonus** — in modal stages (building and peak), an additional logit boost is applied to P4 (perfect 4th, 5 semitones) and P5 (perfect 5th, 7 semitones) intervals, scaled by a `modal_strength` value set per stage (0.0 at sparse/resolution, 0.6 at building, 1.0 at peak, 0.4 at recapitulation). This reflects the quartal and pentatonic character of modal jazz — the stacked-4ths vocabulary of McCoy Tyner, Herbie Hancock, and Wayne Shorter — which is under-represented in the base LSTM's learned distribution but clearly differentiates modal from tonal playing. Statistical analysis of output MIDI confirmed P4 motion rises from ~7% of intervals at modal_strength=0 to ~21% at modal_strength=1.0, matching the expected interval profile for modal jazz.

**Stochastic performance thinning** — at the MIDI output stage only, some short notes are randomly dropped before being sent to the instrument. The full generated phrase is already stored in memory and used for motif detection, arc feedback, register-contrast tracking, and self-play seeding — so the system's internal musical intelligence sees the complete intended phrase. Only what the audience hears is thinned. The effect models the small imprecisions of human articulation: a player running a fast passage will occasionally clip a 16th note, especially under pressure or at peak activity. Drop probability scales linearly with note brevity — notes at or above a quarter note (configurable via `THIN_THRESHOLD_BEATS`, default 0.5 beats) are immune; a note of zero duration would have maximum drop probability. The stage schedule reflects musical context:

| Stage | Max drop probability | Rationale |
|-------|----------------------|-----------|
| sparse | 0% | Every note is precious when material is thin |
| building | 10% | Light thinning as runs begin to appear |
| peak | 15% | Most benefit — fastest passages, most 16th notes |
| recapitulation | 8% | Lyrical return; long notes dominate anyway |
| resolution | 5% | Near silence; almost everything is protected |

The first and last pitched notes of every phrase are always protected regardless of duration — dropping the opening note sounds like a missed entry, and losing the final resolution note is harmonically damaging.

**Phrase breathing** — after the LSTM sampling loop, silence sentinels (REST_PITCH = −1) are spliced into the generated phrase with a bell-curve probability distribution: zero at the very start and end of the phrase, peaking at the midpoint (max 15% chance per inter-note gap). Rest duration is drawn from {0.5, 1.0} beats. The MidiOutput layer translates sentinels to `time.sleep()` calls with no MIDI note sent. This models the breathing and phrasing pauses a human saxophonist would naturally insert — continuous eighth-note streams without gaps are musically inauthentic regardless of harmonic accuracy. Together with stochastic performance thinning (see above), these two mechanisms operate at complementary scales: thinning removes individual short notes within a run, while phrase breathing inserts longer silences between melodic fragments, giving the overall phrase a sense of shape and physical breath.

**Singable duration bias** — the LSTM's training data is dominated by 8th and 16th notes, giving it a strong prior toward fast, busy output. To counter this, a bell-curve logit boost is applied to the duration token vocabulary, centred on the quarter-note token (≈ 0.95 beats) with a width of ±4.5 tokens, pulling sampling toward the 0.5–1.7 beat range where sustained, melodic, "singable" lines live. The boost is scaled by `(1 − rhythmic_density)`, so it is strongest in lyrical stages (full strength at resolution, density=0.1) and progressively suppressed toward the busiest stage (density=0.9 at peak). The result is that mid-register quarter-note lines dominate at sparse and resolution stages while the peak can still drive faster runs.

**Rhythmic density** — the arc controller emits a `rhythmic_density` value (0.0 = lyrical, 1.0 = bebop) for each stage, which the generator uses to scale the singable-duration bias inversely. This creates a natural busyness gradient across the arc: sparse and resolution feel spacious and melodic; peak is the most rhythmically active.

**Rhythmic complementarity** — the stage-based density is blended (40 %) with the reactive complement of the bass phrase's note density. A dense bass phrase (many notes/second) nudges the sax toward a sparser, more lyrical response; a sparse bass phrase nudges the sax toward busier output. This mimics the way jazz musicians trade density to create textural balance — a walking-bass passage invites long sax tones, while a bassist's terse two-note phrase invites a flowing sax run. The arc's macro shape remains in control (60 % weight) so the piece still follows its planned structure.

| Stage | Arc density | Reactive blend | Character |
|-------|-------------|----------------|-----------|
| sparse | 0.2 | 40% of (1 − bass density) | Open, sustained — establish motifs slowly |
| building | 0.5 | 40% of (1 − bass density) | Mixed — lines begin to flow |
| peak | 0.9 | 40% of (1 − bass density) | Fast and active — maximum rhythmic intensity |
| recapitulation | 0.3 | 40% of (1 − bass density) | Lyrical return — recall themes with space |
| resolution | 0.1 | 40% of (1 − bass density) | Slowest — long tones, fading out |

**Register contrast** — the sax is biased toward the register opposite to the bass, so call and response occupy different tonal spaces. When the bass mean pitch is above C4 (MIDI 60) the sax is nudged downward via logit bias; when the bass is below C4 the sax is nudged upward. The bias is smooth and linear (centred on C4, proportional to distance, clamped at ±1.5 logits), so it nudges the output rather than forcing it. The strength scales with the arc stage and is modulated by the bass phrase's ambitus: a wide-ranging bass line reduces the contrast effect (less headroom for register separation), while a narrow bass phrase allows the full contrast to apply. During the sparse stage the contrast is off (the dialogue is still being established); it is strongest during recapitulation, where registral separation reinforces the return of themes.

| Stage | Register contrast strength | Effect |
|-------|---------------------------|--------|
| sparse | 0.0 | Off — dialogue being established |
| building | 0.5 | Moderate — voices begin to separate |
| peak | 0.3 | Reduced — registers intentionally collide for density |
| recapitulation | 0.6 | Strongest — clear registral dialogue on thematic return |
| resolution | 0.4 | Moderate — spacious separation as the piece fades |

**Phrase length control** — four cooperating mechanisms manage phrase length. A duration-token penalty tensor applies a graduated negative logit bias to tokens above ~2 beats. A bell-curve singable-duration boost raises the floor toward quarter notes (see above). A beat accumulator hard-stops generation once the total accumulated beats reaches `max_phrase_beats` (default `MAX_PHRASE_BEATS = 16`). The arc controller caps `n_notes` at 14, uses stage-scaled multipliers (1.1× / 0.75× for leading/following), and enforces a per-stage minimum phrase length floor (4–8 notes) so even the shortest "following" phrase has enough notes to be musically complete.

**Beat-matching / trading bars** — pass `--trade` to enable beat-matching mode. The sax response is capped to the same number of beats as the incoming bass phrase, so trading 2s, 4s, or 8s emerges naturally from however long the bassist chooses to play. A bassist who plays a tight 2-bar phrase (8 beats at 120 BPM) gets an 8-beat sax response; a long 8-bar phrase gets a matching 8-bar response. A minimum floor (`TRADE_BEATS_MIN = 2.0`) prevents very short fills from producing unmusically brief responses. The beat cap is computed from the detected tempo, so it tracks tempo changes automatically. The console log shows `cap=Nb` when trade mode is active. In self-play mode `--trade` creates stable back-and-forth exchanges of equal length.

**Repetition control** — a growing logit penalty is applied to the most recently played pitch token, starting at −2.5 logits on the first immediate repeat and adding −2.0 for each further consecutive repeat. After three same-pitch notes in a row the penalty reaches −6.5 logits, effectively eliminating a fourth repeat while still permitting occasional passing-tone ornaments. The stepwise bias strength was simultaneously reduced (0.4 → 0.1) to remove a partial cancellation that was blunting the penalty's effect.

**Sax riff / insistence** — pass `--sax-riff-prob P` to give the sax a probability P of replaying its previous phrase verbatim instead of generating a fresh response. This is the reverse of the bass riff detection: rather than the sax breaking the bassist's loop, the sax insists on its own phrase and invites the bassist to develop underneath. After `SAX_RIFF_EVOLVE_THRESHOLD` (2) consecutive replays the sax shifts to development mode — it generates a *variation* of its repeated phrase with boosted motivic strength and a directed contour, so the sax is heard to evolve its insistence rather than loop indefinitely. The console logs `[sax riff ×N]` on each replay and `[sax riff ×N develop ...]` when development kicks in. Works in both live-bass and self-play modes.

During the peak stage replays are capped at 1 (rather than the usual 2) to limit harmonic drift: the arc controller advances its internal chord progression on every phrase call regardless of whether the sax replays, so a longer riff cycle at peak silently skips steps in the ii-V-I sequence. One skipped step is tolerable; more would cause the next fresh phrase to land on the wrong chord.

On every riff replay the chord hint (if active) repeats the chord from the phrase's original generation rather than the freshly-advanced harmony, so the same melody always sounds against the same chord. The arc controller's internal state still advances, but this is not exposed to the listener until the riff cycle ends and a fresh phrase is generated.

On every riff replay, each note receives an independent ±8 MIDI velocity jitter (random, applied at output only) so consecutive repetitions feel like a live player re-articulating the phrase rather than a loop. The stored phrase used for musical decision-making is untouched.

**Proactive mode** — the sax does not always wait for a bass phrase to end. When the bassist is sparse or silent, the sax initiates. During the resolution stage, the sax always plays the final phrase. The proactive trigger checks every 0.5 seconds; during the sparse stage the sax will initiate after 7.5 seconds of bass silence. If you want to play first, a short two-note figure followed by one second of silence is enough to trigger the first response before the 7.5-second window expires.

Three additional triggers create more responsive interaction in later stages:

- **Opening fire** (`PROACTIVE_OPENING_DELAY = 3 s`) — with `--auto-start`, the very first sax phrase fires unconditionally after 3 seconds regardless of bass activity. This means Wolfson opens the performance even if there is background noise or pickup bleed on the bass input that would otherwise prevent the silence-based triggers from firing.
- **Extended-riff interrupt** (`PROACTIVE_MAX_WAIT_BEATS = 8`) — in building, peak, and recapitulation stages, if the sax has been silent for more than 8 beats (≈ 8.7 s at 55 BPM, ≈ 5.3 s at 90 BPM) the sax breaks in proactively, creating a trading-bars effect.
- **Active trading** (`PROACTIVE_ACTIVE_INTERVAL_BEATS = 4`) — while the bass is continuously playing, the sax re-fires every 4 beats (≈ 4.4 s at 55 BPM, ≈ 2.7 s at 90 BPM). This sustains a continuous conversation during extended riff sections rather than waiting for the bassist to stop.

**Live phrase peeking** — when the proactive trigger fires while the bass is still mid-phrase (the common case for long phrases), `get_proactive_params()` uses whatever bass notes have accumulated so far rather than falling back on stale features from the last *completed* phrase. With ≥ 4 live notes the system analyses their pitch classes, density, contour, and energy profile in real time, so the sax response reflects the bassist's current material even before they stop playing. Fewer than 4 notes falls back to the last completed phrase features (or the arc defaults if the bass has never completed a phrase).

**Beat-synchronised entry** — when the proactive sax fires while the bass is actively playing, the phrase start is aligned to the next bass beat boundary. `BeatEstimator.last_onset_time` gives the timestamp of the most recent bass note; the sax sleeps for the remainder of the current beat before calling `_respond()`, so it enters on the beat rather than mid-bar. The wait is at most one beat (e.g. 1 second at 60 BPM). If the bass has been silent for more than two beats the grid is considered stale and the sax starts immediately.

**Turn-taking** — the proactive trigger uses `time_since_bass` measured from the most recent individual note, not from the end of the previous complete phrase. Without this, a 15-second bass phrase would appear to the arc controller as 15 seconds of silence, causing the sax to fire proactively mid-phrase. The fix: `arc.touch_bass()` is called on every note_on, keeping `time_since_bass` current. Additionally, the peak and resolution stage proactive rules now require `time_since_bass > PROACTIVE_MIN_INTERVAL` (2 s) before the sax can initiate — so the sax only speaks when the bass has actually paused, not just because enough time has passed since the sax last played.

**Loop mode** — pass `--loop` to restart the arc automatically at the end of each 5-minute performance. PhraseMemory is cleared and the ArcController and HarmonyController are reset between loops, so each arc is a clean slate — appropriate for installations where different users may play in succession. The end-of-arc performance summary is shown before the gap, then the live view resumes automatically. `--loop-gap` sets the pause between arcs (default 8 seconds). In self-play mode the seed phrase re-fires after the gap; in live mode the arc waits for your first bass note.

**Arc timing** — the 5-minute performance arc starts when you play your first bass phrase, not when the script is launched. This means you can start the script (and the web server) well in advance of the performance without consuming arc time — the system waits silently in a pre-show state until the first note arrives. In self-play mode the arc starts immediately as the bootstrap phrase fires automatically. With `--auto-start` the arc also starts immediately on launch, enabling Wolfson to open the performance proactively before you play.

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

### Input filtering

Three filters in `config.py` clean up signals from pitch-to-MIDI converters before they reach the phrase detector:

```python
MIDI_PITCH_MIN    = 28    # E1 — discard notes below open E string
MIDI_PITCH_MAX    = 84    # C6 — discard notes above practical bass range
MIDI_VELOCITY_MIN = 30    # discard very quiet ghost notes (sympathetic resonance)
MIDI_MIN_NOTE_DUR = 0.05  # 50 ms — discard sub-50ms glitch notes
```

The duration filter (`MIDI_MIN_NOTE_DUR`) is particularly important for hardware pitch trackers: analysis of live recordings shows that converters such as the Sonuus i2M produce spurious notes as short as 1–10 ms during pitch transitions, especially on large intervals. Without filtering, these create false phrase boundaries and cause the sax to enter mid-phrase. 50 ms removes the artefacts without affecting any intentional bass playing.

### Warm-up and setup verification

`tools/echo_bass.py` replays each detected bass phrase note-for-note on the sax output channel, with the original pitches, velocities and timing. No model loading is required. It is useful for:

- Confirming MIDI input and output routing end-to-end before starting `main.py`
- Warming up and checking that phrase detection is working correctly
- Practising precise articulation (the echo makes timing and note-end clarity immediately audible)

```bash
python tools/echo_bass.py                      # default settings from config.py
python tools/echo_bass.py --silence 2.0        # longer gap needed to end a phrase
python tools/echo_bass.py --delay 1.0          # pause before replaying (call-and-response feel)
python tools/echo_bass.py --transpose 12       # echo an octave up
```

Port indices, pitch range, velocity threshold and duration filter are all read from `config.py`. The startup output lists available ports with arrows marking the configured ones.

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
python main.py --auto-start     # sax plays proactively before bassist starts
python main.py --sax-riff-prob 0.4   # sax sometimes insists on a phrase; bassist develops underneath
python main.py --trade          # beat-matching: sax matches bass phrase length
python main.py --loop           # loop continuously: new arc starts after each 5 minutes
python main.py --loop --loop-gap 15   # 15-second pause between arcs (default: 8s)
python main.py --chord-hint     # play voiced chord on channel 3 each phrase
python main.py --temperature 0.2   # more adventurous; -0.2 for more conservative
python main.py --motif-displacement   # shift motif injection point by 0.5 or 1.0 beats each phrase
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
- **Waiting screen** — a full-screen "performance will begin shortly" overlay is shown until the first phrase fires. Audience members who arrive early (or whose browser has a stale page from a previous run) see this screen; it disappears automatically when you play your first note.
- **Arc progress bar** — five colour-coded segments (sparse grey, building cyan, peak red, recapitulation green, resolution yellow); each segment fills as the performance progresses with the current stage partially highlighted
- **Stage name** in large type, coloured by stage, with time remaining
- **BPM** and phrase count as large numbers
- **Harmony, scale, contour, velocity** as info cards
- **Last phrase note names** as coloured chips (in the current stage colour)
- **Trigger indicator** — `⟵ bass phrase` or `◎ sax initiates`
- **Pulse animation** — a brief coloured border flash on every new phrase
- **Performance summary** — at arc completion a full-screen overlay replaces the live view with a two-column table comparing the human and algorithm across phrases played, note count, mean note length, short%, pitch range, and dynamic range. Plain-text observations are appended for any metric that falls outside a normal range (e.g. narrow dynamic range, very short notes, large phrase-count imbalance). Framed as observations rather than judgements.
- Auto-reconnects silently if the connection drops

The page is fully self-contained (no CDN dependencies) and loads instantly on a slow venue connection. State is delivered via 2-second polling — robust through any proxy or Cloudflare tunnel.

**Performance workflow** — start the script a few minutes before the audience arrives so the URL is live and the tunnel is connected. The waiting screen holds until you play; the arc clock does not start until your first bass phrase. Share the URL (or display it on screen) and let the audience connect at their own pace before you begin. At the end of the arc the performance summary replaces the live view automatically — the same summary also prints to the console.

##### Public tunnel (eduroam / institutional networks)

On eduroam or other networks where you cannot act as an access point, use `--tunnel` to open a Cloudflare tunnel. This makes an outbound HTTPS/TCP connection to Cloudflare's edge — no inbound ports required. eduroam blocks UDP so the tunnel uses HTTP/2 over TCP (port 443) rather than QUIC.

**Install cloudflared:**
```bash
brew install cloudflare/cloudflare/cloudflared
```

###### Option A — Named tunnel (recommended): stable URL, one-time setup

Requires a free Cloudflare account and a domain managed by Cloudflare.

```bash
# One-time setup
cloudflared tunnel login                        # authorise your domain
cloudflared tunnel create wolfson               # creates credentials file
cloudflared tunnel route dns wolfson wolfson.yourdomain.com
```

Then set in `config.py`:
```python
CLOUDFLARE_TUNNEL_NAME     = "wolfson"
CLOUDFLARE_TUNNEL_HOSTNAME = "wolfson.yourdomain.com"
```

Run:
```bash
python main.py --self-play --web --web-port 8080 --tunnel
```

Startup output:
```
Audience display (local):   http://192.168.1.42:8080
Audience display (stable):  https://wolfson.yourdomain.com
  This URL is permanent — share it before the performance.
```

`wolfson.yourdomain.com` never changes — print it on a programme or slide.

###### Option B — Quick tunnel: no setup, random URL each run

No account or domain needed. URL changes every run.

```bash
python main.py --web --web-port 8080 --tunnel
```

```
Audience display (local):   http://192.168.1.42:8080
Audience display (public):  https://curious-fox-amazing.trycloudflare.com
```

Share the printed URL with the audience at the start of the performance.

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

#### Chord hint

Pass `--chord-hint` to play a short voiced chord on a separate MIDI channel each time the harmony changes — once per phrase (whether triggered by a bass phrase or by the sax playing proactively), at the moment the sax is about to respond. This makes the internal harmonic state directly audible during live testing, like a pianist lightly comping the chord at the start of each exchange.

```bash
python main.py --chord-hint                      # chord hints on MIDI channel 3 (default)
python main.py --chord-hint --comp-channel 4     # use channel 4 instead
python main.py --self-play --chord-hint          # works in self-play too
```

The chord is voiced as a 4-note chord in the C3–F4 register. In tonal sections (peak and resolution) tertian voicings match the chord quality:

| Quality | Voicing | Example |
|---------|---------|---------|
| major | R  3  5  maj7 | Cmaj → C3 E3 G3 B3 |
| dominant | R  3  5  b7 | G7 → G3 B3 D4 F4 |
| minor | R  b3 5  b7 | Dm → D3 F3 A3 C4 |
| diminished | R  b3 b5 bb7 | Bdim → B3 D4 F4 Ab4 |

In modal sections (building and recapitulation) the chord is voiced as four stacked perfect 4ths (R P4 m7 P11 — the quartal stack D G C F for a D root), reflecting the harmonic ambiguity of Dorian/Phrygian playing. A tertian minor-7th chord pulls toward a tonal resolution; a quartal stack is quality-neutral and floats without resolving, matching the modal character of those stages.

The chord sounds for 1.5 beats at velocity 70 then releases. No chord is emitted during the sparse opening stage (the system is in free/chromatic mode with no harmonic target). Route the hint channel to a piano or pad voice in your DAW — a Fender Rhodes works particularly well in the modal sections.

#### Auto-start mode

By default the arc starts lazily on the first bass phrase — Wolfson waits for you to play before it does anything. Pass `--auto-start` to invert this: the arc begins immediately on launch and the sax plays proactively as if it were a musician warming up, before you've touched the bass. Once you join in, the normal reactive/proactive interplay takes over.

```bash
python main.py --auto-start
python main.py --auto-start --chord-hint   # with chord hints
```

This is useful for live performance contexts where the audience is watching from the moment Wolfson starts — it opens the show rather than waiting in silence. The first proactive phrase fires after `PROACTIVE_OPENING_DELAY` (3 seconds) with a neutral seed; as soon as you respond, Wolfson has real material to work with and subsequent phrases develop from the emerging musical conversation.

#### Self-play mode

Pass `--self-play` to run Wolfson autonomously — no MIDI input hardware needed. The sax feeds its own output back as input, creating a continuous generative loop. The 5-minute structural arc still governs the performance.

```bash
python main.py --self-play                          # 120 BPM
python main.py --self-play --bpm 90                # set tempo
python main.py --self-play --dashboard             # with full-screen display
python main.py --self-play --loop                  # loop continuously (new arc after each 5 min)
python main.py --self-play --loop --loop-gap 10 --web   # installation mode
python main.py --self-play --bpm 90 --riff-prob 0.6    # bass riff simulation
```

The system seeds itself with a short D minor pentatonic motif, then responds to each phrase it generates. Each response seeds the next, so the musical conversation develops and evolves over the full arc. Useful for:
- Listening to the model's musical character without a musician present
- Testing the arc structure and harmonic progression in real time
- Leaving it running as a generative ambient piece
- Installation contexts where the system should run unattended and reset automatically
- Testing how the system responds to a repeating bass riff (`--riff-prob`)

**Riff / ostinato simulation (`--riff-prob`)** — by default the self-play loop always uses the latest sax output as the next bass phrase, producing pure lick trading. With `--riff-prob P` (0.0–1.0), each feedback cycle has probability P of re-injecting the *previous* bass phrase instead, simulating a repeating riff or ostinato. At `--riff-prob 0.6` the bass typically repeats one or two times before changing; at `1.0` the bass never changes (pure ostinato). Once the same phrase has appeared **3 times in a row**, the sax enters development mode: motivic strength is boosted and the contour is pushed away from neutral, so the response clearly evolves rather than trading the same lick back. The console logs `[riff ×N ...]` whenever this threshold is crossed. Riff state resets cleanly between `--loop` iterations. Note that in self-play the repeated bass phrases are **not audible** as a separate voice — they are internal input state only. What you hear is the single sax output line; the effect of the riff simulation is in how that line develops melodically in response to the repetition.

**Two-channel dialogue** — phrases automatically alternate between MIDI channel 1 and MIDI channel 2, making the call-and-response structure directly visible in a DAW and directly audible if the two channels are routed to different sounds.

| Channel | Phrases | Suggested voice |
|---------|---------|-----------------|
| 1 | Odd (1, 3, 5 …) | Alto sax / higher register |
| 2 | Even (2, 4, 6 …) | Tenor sax / lower register |
| 3 | Chord hints (`--chord-hint`) | Piano or pad |

DAW setup (Logic, Ableton, etc.):
1. Create two software instrument tracks, both receiving from the Wolfson MIDI output port
2. Set one track to receive channel 1, the other to channel 2
3. Assign different sounds — contrasting timbres (alto + tenor, oboe + clarinet) make the dialogue most audible
4. Record-arm both tracks — the call-and-response appears as two separate MIDI regions
5. Optionally add a third track on channel 3 with a piano or pad sound for chord hints (`--chord-hint`)

The channel numbers are configurable in `config.py` (`SELF_PLAY_CH_A`, `SELF_PLAY_CH_B`). In live-bass mode the sax always plays on channel 1.

Console output logs `ch=N` and `chord=Xqual` (e.g. `chord=Dm`, `chord=G7`, `chord=NC`) on every phrase so you can see which voice is speaking and what harmony is active. A rolling statistics block is printed every 8 phrases. Use `--dashboard` for the full-screen display.

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

### Generator tests

`tests/run_tests.py` runs 21 feature tests programmatically — no MIDI hardware needed. Each test constructs a synthetic bass phrase, analyses it, generates a sax response with explicit parameters, and writes a log file. A combined two-track demo MIDI is also produced.

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

### Unit tests

`tests/test_arc_controller.py` and `tests/test_phrase_detector.py` provide unit tests for the real-time state machines. No MIDI hardware or model loading required — all note events are injected with synthetic timestamps, and the arc controller's timing state is manipulated directly.

```bash
~/.pyenv/versions/3.12.1/bin/python3 -m pytest tests/test_arc_controller.py tests/test_phrase_detector.py -v
```

**Arc controller tests (12)** — cover `touch_bass()`, `PROACTIVE_MIN_INTERVAL` gate, peak-stage bass-activity guard, resolution-stage bass-activity guard, building-stage silence trigger, and arc-not-started guard.

**Phrase detector tests (20)** — cover basic phrase completion, ghost-note silence-timer isolation (the fix for premature phrase endings on guitar/i2M), stale note_off handling (the fix for watchdog cancellation), watchdog for sustained notes, monophony, and sub-minimum duration filtering. A short `silence_threshold` (60 ms) is used so timer-dependent tests complete in under 2 seconds. Also covers four hardware MIDI scenarios:

| Scenario | Description |
|----------|-------------|
| i2M note-extend | Stale note_offs arrive after the next note_on (as produced by the Sonuus i2M in note-extend mode). Tests that stale note_offs do not reset the silence timer or split a legato run into separate phrases. |
| i2M re-trigger | The i2M fires a note_on then an immediate note_off for the same pitch (zero duration) when re-triggering a held note. The zero-duration note is filtered by `MIDI_MIN_NOTE_DUR`; the silence timer is allowed to start when `self._timer is None` (cancelled by the re-triggering note_on), preventing the phrase from being stranded. Two tests: phrase fires correctly after the ghost note_off, and the timer is cancelled when the next real note arrives quickly. |
| Missing note_offs | Hardware that never sends note_off events. Monophony closes all notes except the last; the watchdog timer closes the last note after the silence threshold. All notes land in one phrase. |
| Velocity-zero note_off | `note_on` with velocity 0 is the standard MIDI encoding for note_off. Tests that `MidiListener` routes these correctly — treating them as note_offs rather than silent note_ons — and that they complete a phrase end-to-end. |

## Analysis tools

`tools/analyse_midi.py` parses self-play MIDI recordings and prints per-stage note duration statistics for the melody channel, used to verify that the arc creates audible stage differentiation at different tempos and riff probabilities.

```bash
# Single file
python tools/analyse_midi.py /path/to/recording.mid

# Side-by-side comparison across tempos or runs
python tools/analyse_midi.py 60bpm.mid 90bpm.mid 120bpm.mid

# Time-series plot (requires matplotlib)
python tools/analyse_midi.py a.mid b.mid --plot
python tools/analyse_midi.py a.mid b.mid --plot --plot-out comparison.png
```

For each stage (sparse / building / peak / recapitulation / resolution) it reports note count, mean duration, median duration, short% (< 0.4 beats) and long% (≥ 0.75 beats). A summary comparison table is printed when multiple files are given.

`--plot` generates a three-panel PNG showing rolling-window curves (40-second window, 10-second step) plotted against the full 5-minute arc time axis, with stage regions colour-shaded and one line per input file:

- **Mean note duration** (beats) — lyrical quality; rises toward the peak stage as the arc drives longer motifs
- **Short note %** — busyness; inversely mirrors mean duration
- **Note rate** (notes/min) — absolute activity level across the arc

Multiple files on the same plot makes it straightforward to compare different tempos, different `--riff-prob` settings, or repeated runs at the same tempo to see how much the output varies.

## Project structure

```
wolfson/
├── main.py                       Entry point (full 5-minute performance); stochastic thinning, phrase-shape dynamics, loop mode
├── demo.py                       Feature-focused testing without the arc
├── test-midi-in.py               Verify MIDI input port and event routing
├── osc-monitor.py                Print incoming OSC messages (test OSC output)
├── config.py                     All tunable parameters (incl. SELF_PLAY_CH_A/B, TRADE_BEATS_MIN)
├── wolfson_train.ipynb           Google Colab training notebook
├── requirements.txt
├── input/
│   ├── midi_listener.py          MIDI input, note events; calls arc.touch_bass() on every note_on
│   ├── phrase_detector.py        Segments note stream into phrases; ghost-note and stale-note_off guards
│   ├── phrase_analyzer.py        Phrase features: contour, density, Q&A type, swing, dynamics, energy profile, interval motifs, lyrical motifs
│   └── beat_estimator.py         Live tempo estimation from bass onsets; last_onset_time for beat-sync
├── memory/
│   └── phrase_memory.py          Stores phrases + motifs + lyrical motifs for recall and development
├── generator/
│   ├── lstm_model.py             LSTM with chord conditioning
│   ├── phrase_generator.py       Seeds LSTM; contour, scale, swing, energy arc, motif, voice leading, modal leap, register contrast, singable duration bias, rest injection, phrase length control
│   └── train.py                  Training script
├── controller/
│   ├── arc_controller.py         Arc, leadership, proactive mode (with bass-activity guard), touch_bass(), opening echo (sparse stage), modal_strength + rhythmic_density (with reactive complementarity) + register_contrast + swing range schedules, lyrical motif recall
│   └── harmony.py                Harmonic modes: free, modal, progression, pedal
├── output/
│   ├── midi_output.py            Per-note MIDI playback; queue architecture with single output thread eliminates stuck-note races; on_complete callback defers arc bookkeeping to playback completion
│   ├── dashboard.py              Rich full-screen terminal display, black background, high-contrast (--dashboard)
│   ├── web_display.py            Audience web display served over HTTP with 2-second polling (--web); end-of-arc performance summary overlay
│   └── osc_output.py             UDP/OSC phrase events for stage visuals (--osc-host)
├── data/
│   ├── encoding.py               Pitch+duration token encoding
│   ├── chords.py                 Chord parsing and 49-token vocabulary
│   ├── scales.py                 Mode interval tables and scale pitch-class helpers
│   ├── instruments.py            Instrument family definitions and pitch ranges
│   └── prepare.py                WJD data preparation script
├── tests/
│   ├── run_tests.py              Generator test suite; writes logs + demo.mid
│   ├── test_arc_controller.py    Unit tests: proactive logic, turn-taking guards, touch_bass()
│   ├── test_phrase_detector.py   Unit tests: phrase completion, ghost notes, stale note_offs, watchdog
│   └── logs/                     Per-test log files (generated)
├── tools/
│   ├── echo_bass.py              Echo bass phrases on sax output for setup verification and warm-up
│   └── analyse_midi.py           Stage-by-stage duration analysis of self-play MIDI recordings
└── docs/
    ├── wolfson.pdf               Presentation slides (PDF, viewable on GitHub)
    ├── wolfson.pptx              Presentation slides (editable source)
    ├── architecture.pdf          System architecture diagram
    ├── performance_plan.pdf      Single-page performer cue sheet (landscape A4, by arc stage)
    ├── performance_plan.tex      LaTeX source for the performer cue sheet
    ├── dashboard_guide.pdf       Dashboard display guide
    ├── programme_note.md         Audience programme note
    └── rig.md                    Hardware and DAW setup (i2M, Logic Pro, MIDI routing)
```

## Extending to other instruments

```bash
python data/prepare.py --instrument trumpet
python generator/train.py --instrument trumpet
```

Set `DEFAULT_INSTRUMENT = "trumpet"` in `config.py`, or pass `instrument="trumpet"` to `PhraseGenerator`. Families and pitch ranges are in `data/instruments.py`.

## Presentation

A short overview of the Wolfson project: [`docs/wolfson.pdf`](docs/wolfson.pdf)

## Acknowledgements

Solo transcriptions from the [Weimar Jazz Database](https://jazzomat.hfm-weimar.de/), Jazzomat Research Project, Hochschule für Musik Franz Liszt Weimar.

Wolfson was created for a performance at [Wolfson College, Oxford](https://www.wolfson.ox.ac.uk/) on 1st May 2026.

It draws on insights from previous work with many musicians and computer scientists, in particular those involved with [Music, Computing and AI — Three Talks and a Mini-Concert](https://www.stcatz.ox.ac.uk/music-computing-and-ai-three-talks-and-a-mini-concert/) at St Catherine's College, Oxford, on 20th March 2024: Professor Ray d'Inverno (University of Southampton), Professor Mark d'Inverno (Goldsmiths College, University of London), and Professor David De Roure (University of Oxford), who performed jazz standards together as bass, piano, and drums.

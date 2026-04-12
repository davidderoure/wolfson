# Wolfson — Rig Setup

## Hardware

- **Sonuus i2M** connected to Mac via USB and configured using sonuus desktop app:
  - Output set to **MIDI channel 2** 
  - Instrument is 4 String Bass
  - Note Extend: **on**
  - Can set gate to constrain register (but this is handled by wolfson)

- Bass audio also captured via i2M audio output (Logic track 4)

## Logic Pro session

| Track | Instrument | MIDI input | Channel |
|---|---|---|---|
| 1 | Upright / bass | i2M | ch 2 |
| 2 | Tenor sax | IAC Driver Bus 2 | ch 1 |
| 3 | Chord hint (Fender Rhodes) | IAC Driver Bus 2 | ch 3 |
| 4 | Audio | i2M audio input | — |

Track 1 is **muted** — records the raw bass MIDI for testing and analysis but does not play back. If playing the bass software instrument live, use Velocity Limit (e.g. 30-127) and Key limit (e.g. E0 to G3) in Logic track to exclude ghost notes (or create another track with the i2M ch 2 input to do this).

Logic instruments:

- Bass - multiple instruments available in Logic library e.g. 
Studio Upright Bass.

- Comp - suggest Fender Rhodes. Multiple organs available in stock Library. Using Vintage Keys -> Vintage Electric Piano and selected Classic. Adjust chorus etc also tine bell (could use velocity processor if too much tine). 

- Sax - Wolfson is tuned to Tenor Sax but beware register and tuning of sampled instruments may not match. Stock woodwind Horns -> tenor sax is safe. Studio Tenor Sax is limited (misses Eb4 and above in Logic notation - unless you select "extended keyrange" in options at bottom).

## Wolfson configuration

MIDI Channels preset in main.py with Sax on 1, Bass on 2, Comp on 3

- `MIDI_INPUT_PORT = 2` in `config.py` (i2M)
- `MIDI_VELOCITY_MIN = 30` in `config.py` — filters i2M ghost notes (vel ≤ 25) before they reach the phrase detector
- Wolfson outputs sax (ch 1) and chord hints (ch 3) to **IAC Driver Bus 2**, which Logic reads on tracks 2 and 3

## Signal flow

```
Bass
 └─► i2M ──┬──► USB/MIDI ──► Wolfson (MIDI_INPUT_PORT=2, vel filter ≥30)
            │                      └──► IAC Driver Bus 2 ──► Logic tracks 2 & 3
            └──► Logic track 1 (ch 2, muted — for recording/analysis)
            └──► Logic track 4 (audio)
```

## Testing

Use `tools/echo_bass.py` to verify i2M is being received correctly before a session:

```bash
python3 tools/echo_bass.py
```

Input should show as the i2M device.

For bench testing, a MIDI keyboard can be plugged into USB and check where it appears in echo_bass, then modify MIDI_INPUT_PORT in config.py accordingly.  Set keyboard down two octaves. A 24 fret 4 string bass goes from MIDI 28 to 67 (E1 to G4, or E0 to G3 in the Logic convention).


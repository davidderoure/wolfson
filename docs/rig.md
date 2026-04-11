# Wolfson — Rig Setup

## Hardware

- **Sonuus i2M** connected to Mac via USB
  - Output set to **MIDI channel 2**
  - Note Extend: **on**
- Bass audio also captured via i2M audio output (Logic track 4)

## Logic Pro session

| Track | Instrument | MIDI input | Channel |
|---|---|---|---|
| 1 | Upright / bass | i2M | ch 2 |
| 2 | Tenor sax | IAC Driver Bus 2 | ch 1 |
| 3 | Chord hint (Fender Rhodes) | IAC Driver Bus 2 | ch 3 |
| 4 | Audio | i2M audio input | — |

Track 1 is **muted** — records the raw bass MIDI for testing and analysis but does not play back.

## Wolfson configuration

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

"""Sends generated sax phrases to a MIDI output port."""

import rtmidi
import time
import threading
from config import MIDI_OUTPUT_PORT, REST_PITCH

# Semitone intervals from root for each chord quality (maj7, dom7, min7, dim7)
_CHORD_VOICINGS = {
    0: (0, 4, 7, 11),   # major  →  R  3  5  maj7
    1: (0, 4, 7, 10),   # dom    →  R  3  5  b7
    2: (0, 3, 7, 10),   # minor  →  R  b3 5  b7
    3: (0, 3, 6,  9),   # dim    →  R  b3 b5 bb7
}

DEFAULT_VELOCITY    = 80
ARTICULATION_RATIO  = 0.85   # note sounds for this fraction of its slot; rest is silence
MAX_NOTE_DUR        = 5.0    # hard cap (seconds) — prevents runaway stuck notes if the
                             # LSTM emits a max-length token near the 55 BPM floor


class MidiOutput:
    def __init__(self):
        self._midi_out  = rtmidi.MidiOut()
        # Exclusive-playback sequencing: each call to play_phrase increments
        # _play_seq and records its own sequence number.  When a newer call
        # starts it clears the channel and bumps the counter; the older call
        # detects the mismatch and exits, preventing overlapping note_ons that
        # would leave polyphonic voices stuck on the synthesiser.
        self._play_seq  = 0
        self._seq_lock  = threading.Lock()

    def start(self):
        ports = self._midi_out.get_ports()
        if not ports:
            raise RuntimeError("No MIDI output ports found.")
        print(f"MIDI output: {ports[MIDI_OUTPUT_PORT]}")
        self._midi_out.open_port(MIDI_OUTPUT_PORT)

    def stop(self):
        self._midi_out.close_port()

    def _flush_channel(self, ch: int):
        """Send note_off for every pitch on a 0-indexed channel."""
        for pitch in range(128):
            self._midi_out.send_message([0x80 | ch, pitch, 0])

    def play_phrase(
        self,
        pitches:   list[int],
        durations: list[float],          # per-note duration in seconds
        velocity   = DEFAULT_VELOCITY,   # int OR list[int] for per-note dynamics
        channel:   int   = 1,
    ):
        """
        Play a phrase note-by-note with per-note durations.

        velocity may be a single int (applied to all notes) or a list of ints
        (one per note) for per-note dynamic shaping from the energy arc.

        Each note sounds for ARTICULATION_RATIO of its slot duration, with a
        short silence before the next note — giving natural sax articulation.

        Only one phrase plays at a time.  If a second call arrives while this
        one is still running (proactive loop vs bass-response overlap), the new
        call claims the channel: it flushes all notes and increments the
        sequence counter.  The old call detects the counter change at the top
        of its next note iteration and exits cleanly, so no voices are left
        open on the synthesiser.
        """
        ch = channel - 1

        # Claim this playback slot atomically: bump the counter and flush any
        # notes left sounding by a previous (possibly still-sleeping) call.
        with self._seq_lock:
            self._play_seq += 1
            my_seq = self._play_seq
            self._flush_channel(ch)

        active_pitch = None
        try:
            for i, (pitch, dur) in enumerate(zip(pitches, durations)):
                # A newer phrase has started — exit immediately.  The channel
                # was already flushed by the new call's startup, so no orphan
                # notes remain.
                if self._play_seq != my_seq:
                    return
                if pitch == REST_PITCH:
                    # Silence sentinel — just wait, no MIDI output
                    time.sleep(dur)
                    continue
                vel         = velocity[i] if isinstance(velocity, list) else velocity
                vel         = max(1, min(127, int(vel)))
                dur         = min(dur, MAX_NOTE_DUR)
                sound_dur   = max(0.02, dur * ARTICULATION_RATIO)
                silence_dur = max(0.005, dur - sound_dur)
                active_pitch = pitch
                self._midi_out.send_message([0x90 | ch, pitch, vel])
                time.sleep(sound_dur)
                # Check again after sleeping — new phrase may have started
                # during the note.  It will have flushed the channel already,
                # so skip the note_off (it was already sent) and exit.
                if self._play_seq != my_seq:
                    return
                self._midi_out.send_message([0x80 | ch, pitch, 0])
                active_pitch = None
                time.sleep(silence_dur)
        except Exception:
            # Ensure no note is left sounding if playback is interrupted
            if active_pitch is not None:
                self._midi_out.send_message([0x80 | ch, active_pitch, 0])

    def play_chord_hint(
        self,
        chord_idx:    int,
        beat_dur_sec: float,
        channel:      int = 3,
        velocity:     int = 55,
    ):
        """
        Play a voiced 4-note chord on the hint channel in a background thread.

        Fired once per bass phrase to make the internal harmonic state
        directly audible — like a pianist lightly picking out the chord at
        the start of each exchange.  The chord sounds for 1.5 beats then
        releases silently.

        chord_idx    — Wolfson chord index (0–47); NC_INDEX (48) = silent
        beat_dur_sec — seconds per beat at the current tempo
        channel      — 1-indexed MIDI channel (default 3)
        velocity     — MIDI velocity; 55 is a quiet comp
        """
        from data.chords import NC_INDEX, N_QUALITIES
        if chord_idx == NC_INDEX:
            return

        root_pc  = chord_idx // N_QUALITIES        # 0–11
        quality  = chord_idx %  N_QUALITIES        # 0–3
        root_midi = 48 + root_pc                   # C3–B3
        notes    = [root_midi + i for i in _CHORD_VOICINGS.get(quality, _CHORD_VOICINGS[0])]
        dur      = beat_dur_sec * 1.5
        ch       = channel - 1

        def _play():
            for n in notes:
                self._midi_out.send_message([0x90 | ch, n, velocity])
            time.sleep(dur)
            for n in notes:
                self._midi_out.send_message([0x80 | ch, n, 0])

        threading.Thread(target=_play, daemon=True).start()

    def silence(self, channels=1):
        """
        Silence one or more MIDI channels.

        Sends CC 123 (All Notes Off) and CC 120 (All Sound Off) followed by
        explicit note_off for every pitch — Logic software instruments ignore
        the CC messages and only respond to per-pitch note_offs.

        channels may be a single int or a list/tuple of ints, e.g.
        ``silence(1)``, ``silence([1, 2])``.
        """
        if isinstance(channels, int):
            channels = [channels]
        for channel in channels:
            ch = channel - 1
            self._midi_out.send_message([0xB0 | ch, 123, 0])   # All Notes Off
            self._midi_out.send_message([0xB0 | ch, 120, 0])   # All Sound Off
            for pitch in range(128):
                self._midi_out.send_message([0x80 | ch, pitch, 0])

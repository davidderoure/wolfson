"""
OSC output for Wolfson.

Sends phrase events over UDP/OSC to any receiver — TouchDesigner, Max/MSP,
Processing (oscP5), Ableton Live (with Max4Live), etc.

All sends are fire-and-forget (UDP); network errors are silently suppressed
so OSC problems never interrupt the performance.

Address scheme
--------------

Arc / time
  /wolfson/arc/elapsed       f   seconds elapsed in the performance (0–300)
  /wolfson/arc/progress      f   normalised position through the arc (0.0–1.0)
  /wolfson/arc/stage         s   "sparse" | "building" | "peak" |
                                 "recapitulation" | "resolution"
  /wolfson/arc/stage_index   i   0=sparse  1=building  2=peak
                                 3=recapitulation  4=resolution

Tempo
  /wolfson/bpm               f   current tempo estimate (BPM)

Harmony
  /wolfson/harm/mode         s   "free" | "modal" | "progression" | "pedal"
  /wolfson/harm/mode_index   i   0=free  1=modal  2=progression  3=pedal

Scale / tonality tracking
  /wolfson/scale/source      s   "bass" | "blend" | "arc"
  /wolfson/scale/source_f    f   0.0=arc  0.5=blend  1.0=bass

Phrase parameters
  /wolfson/phrase/trigger    s   "bass" | "sax"
  /wolfson/phrase/leadership s   "bass" | "sax"
  /wolfson/phrase/contour    s   "ascending" | "descending" | "neutral"
  /wolfson/phrase/contour_f  f   1.0=ascending  0.0=neutral  -1.0=descending
  /wolfson/phrase/velocity   i   MIDI velocity (40–110)
  /wolfson/phrase/n_notes    i   number of notes in this phrase

Pitches
  /wolfson/pitches           i…  one int argument per note (MIDI pitch 0–127)
"""

from pythonosc import udp_client

# ---------------------------------------------------------------------------
# Lookup tables for numeric encodings
# ---------------------------------------------------------------------------

ARC_TOTAL = 300.0

_STAGE_INDEX = {
    "sparse":          0,
    "building":        1,
    "peak":            2,
    "recapitulation":  3,
    "resolution":      4,
}

_HARM_INDEX = {
    "free":        0,
    "modal":       1,
    "progression": 2,
    "pedal":       3,
}

_SCALE_F = {
    "arc":   0.0,
    "blend": 0.5,
    "bass":  1.0,
}

_CONTOUR_F = {
    "ascending":  1.0,
    "neutral":    0.0,
    "descending": -1.0,
}


# ---------------------------------------------------------------------------
# OscOutput
# ---------------------------------------------------------------------------

class OscOutput:
    """
    Sends Wolfson phrase events over UDP/OSC.

    Parameters
    ----------
    host : str
        IP address of the OSC receiver (default: localhost).
    port : int
        UDP port of the OSC receiver (default: 9000).
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 9000):
        self._client = udp_client.SimpleUDPClient(host, port)
        self._host   = host
        self._port   = port

    def __repr__(self) -> str:
        return f"OscOutput({self._host}:{self._port})"

    def send_phrase(
        self,
        params:       dict,
        notes:        list,
        bpm:          float,
        elapsed:      float,
        triggered_by: str,
    ):
        """
        Send all phrase parameters as individual OSC messages.

        Called once per phrase, before playback begins, so receivers can
        react in sync with the first note.
        """
        try:
            self._send_all(params, notes, bpm, elapsed, triggered_by)
        except Exception:
            pass   # never let OSC errors surface into the audio path

    # -----------------------------------------------------------------------

    def _send_all(
        self,
        params:       dict,
        notes:        list,
        bpm:          float,
        elapsed:      float,
        triggered_by: str,
    ):
        s = self._client.send_message   # shorthand

        stage   = params.get("stage",          "sparse")
        harm    = params.get("harmonic_mode",   "free")
        src     = params.get("scale_source",    "arc")
        contour = params.get("contour_target",  "neutral")
        vel     = int(params.get("velocity",     80))
        lead    = params.get("leadership",      "bass")
        pitches = [n["pitch"] for n in notes]

        # Arc / time
        s("/wolfson/arc/elapsed",      float(elapsed))
        s("/wolfson/arc/progress",     float(min(1.0, elapsed / ARC_TOTAL)))
        s("/wolfson/arc/stage",        str(stage))
        s("/wolfson/arc/stage_index",  int(_STAGE_INDEX.get(stage, 0)))

        # Tempo
        s("/wolfson/bpm",              float(bpm))

        # Harmony
        s("/wolfson/harm/mode",        str(harm))
        s("/wolfson/harm/mode_index",  int(_HARM_INDEX.get(harm, 0)))

        # Scale tracking
        s("/wolfson/scale/source",     str(src))
        s("/wolfson/scale/source_f",   float(_SCALE_F.get(src, 0.0)))

        # Phrase parameters
        s("/wolfson/phrase/trigger",    str(triggered_by))
        s("/wolfson/phrase/leadership", str(lead))
        s("/wolfson/phrase/contour",    str(contour))
        s("/wolfson/phrase/contour_f",  float(_CONTOUR_F.get(contour, 0.0)))
        s("/wolfson/phrase/velocity",   vel)
        s("/wolfson/phrase/n_notes",    int(len(notes)))

        # Pitches — sent as a single message with one int argument per note
        if pitches:
            s("/wolfson/pitches", pitches)

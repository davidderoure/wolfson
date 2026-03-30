# MIDI
MIDI_INPUT_PORT = 0       # index of your bass pitch-to-MIDI interface
MIDI_OUTPUT_PORT = 1      # index of synth/output for the sax voice

# Beat-matching ("trading bars") mode
# When True, the sax response is capped to the same number of beats as the
# incoming bass phrase, so trading 2s, 4s, or 8s emerges naturally from
# however long the bassist plays.  A hard floor (TRADE_BEATS_MIN) prevents
# the sax from generating a phrase too short to be musical if the bass plays
# a very brief fill.  Enabled via --trade on the command line.
TRADE_BEATS_MODE = False
TRADE_BEATS_MIN  = 2.0    # minimum sax phrase length in beats even in trade mode

# Self-play two-channel split
# In --self-play mode the two voices alternate between these MIDI channels
# so they can be routed to different sounds in a DAW (e.g. alto sax + tenor sax).
# In normal (live-bass) mode the sax always plays on SELF_PLAY_CH_A.
SELF_PLAY_CH_A = 1   # "call" voice  — odd phrases  (1, 3, 5 …)
SELF_PLAY_CH_B = 2   # "response" voice — even phrases (2, 4, 6 …)

# Phrase detection
SILENCE_THRESHOLD_SEC = 1.0   # gap that marks a phrase boundary
MIN_PHRASE_NOTES = 2           # ignore micro-phrases shorter than this

# Phrase memory
MAX_PHRASES_STORED = 64

# Generator
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
MAX_GENERATED_NOTES = 16      # max note-pairs (pitch+dur) per sax response phrase
GENERATION_TEMPERATURE = 0.9
DEFAULT_INSTRUMENT = "sax"    # which trained model to load at startup
REST_PITCH = -1                # sentinel: a note dict with this pitch = silence (no MIDI)

# Tempo hint for the beat estimator (BPM).
# Set to your approximate performance tempo if playing above ~160 BPM
# or in an unusual register. 0 = no hint (estimator uses its running average).
TEMPO_HINT_BPM = 0

# Dashboard and OSC (both off by default; enable via CLI flags)
DASHBOARD_ENABLED = False   # --dashboard  : rich full-screen terminal display
OSC_ENABLED       = False   # --osc-host   : OSC UDP output
OSC_HOST          = "127.0.0.1"
OSC_PORT          = 9000

# Structural arc (all times in seconds from performance start)
ARC = {
    "sparse":          (0,    60),
    "building":        (60,   150),
    "peak":            (150,  210),
    "recapitulation":  (210,  270),
    "resolution":      (270,  300),
}

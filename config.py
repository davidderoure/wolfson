# MIDI
MIDI_INPUT_PORT = 0       # index of your bass pitch-to-MIDI interface
MIDI_OUTPUT_PORT = 0      # index of synth/output for the sax voice

# Phrase detection
SILENCE_THRESHOLD_SEC = 0.4   # gap that marks a phrase boundary
MIN_PHRASE_NOTES = 2           # ignore micro-phrases shorter than this

# Phrase memory
MAX_PHRASES_STORED = 64

# Generator
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
MAX_GENERATED_NOTES = 16      # max note-pairs (pitch+dur) per sax response phrase
GENERATION_TEMPERATURE = 0.9
DEFAULT_INSTRUMENT = "sax"    # which trained model to load at startup

# Tempo hint for the beat estimator (BPM).
# Set to your approximate performance tempo if playing above ~160 BPM
# or in an unusual register. 0 = no hint (estimator uses its running average).
TEMPO_HINT_BPM = 0

# Structural arc (all times in seconds from performance start)
ARC = {
    "sparse":          (0,    60),
    "building":        (60,   150),
    "peak":            (150,  210),
    "recapitulation":  (210,  270),
    "resolution":      (270,  300),
}

# MIDI
MIDI_INPUT_PORT = 2       # index of your bass pitch-to-MIDI interface
MIDI_OUTPUT_PORT = 1      # index of synth/output for the sax voice

# MIDI input pitch filter
# Notes outside this range are silently ignored before reaching the phrase
# detector. Useful for suppressing sympathetic string resonance from
# pitch-to-MIDI converters (e.g. Sonuus i2M open-string artefacts).
# Set 0 / 127 to disable filtering entirely.
MIDI_PITCH_MIN = 28   # E1 — lowest intentional bass note (open E string)
MIDI_PITCH_MAX = 84   # C6 — well above practical bass range

# MIDI input velocity filter
# Notes with velocity below this threshold are ignored before reaching the
# phrase detector. Sympathetic string resonance from pitch-to-MIDI converters
# (e.g. Sonuus i2M) typically produces ghost notes at very low velocity.
# Raise this value if spurious notes persist; lower it if quiet intentional
# notes are being dropped. Set to 0 to disable.
MIDI_VELOCITY_MIN = 20

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

# Stochastic performance thinning
# At the MIDI output stage only — the full generated phrase is kept in memory
# for motif detection, arc feedback, and self-play seeding.  Only what is
# actually played is thinned, so the system's internal musical intelligence
# is never degraded.
#
# Notes shorter than THIN_THRESHOLD_BEATS are eligible for dropping.
# Drop probability scales linearly from 0 at the threshold to the stage
# maximum at zero duration, so very short notes are most vulnerable and
# quarter-note-length notes are immune.  The first and last pitched notes
# of every phrase are always protected.
#
# Set THIN_THRESHOLD_BEATS = 0.0 to disable thinning entirely.
THIN_THRESHOLD_BEATS = 0.5   # notes >= this duration (beats) are never dropped

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

# Cloudflare Named Tunnel (optional — stable URL across runs)
# Create once with:
#   cloudflared tunnel login
#   cloudflared tunnel create wolfson
#   cloudflared tunnel route dns wolfson wolfson.numbersintonotes.net
# Then create ~/.cloudflared/config.yml pointing the hostname at localhost:<web-port>
# Leave blank to use quick tunnels (random trycloudflare.com URL each run).
CLOUDFLARE_TUNNEL_NAME     = ""   # e.g. "wolfson"
CLOUDFLARE_TUNNEL_HOSTNAME = ""   # e.g. "wolfson.numbersintonotes.net"

# TinyURL integration (optional)
# When set, Wolfson automatically updates a fixed TinyURL alias to point to
# the current trycloudflare.com URL each run, so the audience always gets
# the same link regardless of which tunnel URL was assigned.
#
# Setup (one-time):
#   1. Create an API token at https://tinyurl.com/app/settings/tokens
#   2. Create your alias once at https://tinyurl.com/create — e.g. "wolfson-live"
#      (tinyurl.com/wolfson-live will then always point to the current tunnel)
#   3. Fill in the values below
TINYURL_TOKEN = ""       # your API bearer token
TINYURL_ALIAS = ""       # just the alias part, e.g. "wolfson-live"

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

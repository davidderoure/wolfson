"""
Chord parsing and encoding for WJD chord strings.

Quality is simplified to four classes that capture the harmonic function
most relevant to melody generation:

  0  major      maj, maj7, maj9, add9, 6, aug (treated as colour of major)
  1  dominant   7, 9, 11, 13, 7b9, 7#9, sus (sus → dominant function)
  2  minor      m, m7, m9, m11, m6
  3  diminished dim, hdim, m7b5, ø

Combined chord index = root_semitone * N_QUALITIES + quality_class   (0-47)
NC_INDEX = 48   (no chord / rest / unknown)
CHORD_VOCAB_SIZE = 49
"""

import re

ROOTS = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

_ROOT_MAP: dict[str, int] = {name: i for i, name in enumerate(ROOTS)}
_ROOT_MAP.update({'C#': 1, 'D#': 3, 'F#': 6, 'G#': 8, 'A#': 10, 'B#': 0, 'Cb': 11})

N_QUALITIES     = 4
N_CHORD_TYPES   = 12 * N_QUALITIES   # 48
NC_INDEX        = N_CHORD_TYPES      # 48
CHORD_VOCAB_SIZE = N_CHORD_TYPES + 1 # 49

QUAL_MAJOR = 0
QUAL_DOM   = 1
QUAL_MINOR = 2
QUAL_DIM   = 3


def parse_chord(s: str) -> int:
    """
    Parse a WJD chord string → chord index (0-48).
    Returns NC_INDEX for no-chord or unparseable input.
    """
    if not s:
        return NC_INDEX
    s = s.strip()
    if s.upper() in ('NC', 'N', 'X', ''):
        return NC_INDEX

    # Extract root (longest match first to handle Db before D, etc.)
    root = None
    rest = s
    for candidate in sorted(_ROOT_MAP, key=len, reverse=True):
        if s.startswith(candidate):
            root = _ROOT_MAP[candidate]
            rest = s[len(candidate):]
            break

    if root is None:
        return NC_INDEX

    quality = _parse_quality(rest)
    return root * N_QUALITIES + quality


def _parse_quality(rest: str) -> int:
    r = rest.lower()

    # Diminished — check before minor ('m7b5' and 'hdim' contain 'm')
    if any(k in r for k in ('dim', 'hdim', 'm7b5', 'ø', 'o7')):
        return QUAL_DIM

    # Major — check before minor ('maj7' starts with 'm')
    if 'maj' in r or r == '' or r.startswith('add') or r.startswith('6'):
        return QUAL_MAJOR

    # Minor — check before dominant (Cm7 starts with 'm' and contains '7')
    if r.startswith('m') or r.startswith('-'):
        return QUAL_MINOR

    # Dominant — plain 7/9/11/13, sus, aug
    if re.search(r'7|9|11|13|sus|\+|aug', r):
        return QUAL_DOM

    return QUAL_MAJOR


def chord_index_to_name(idx: int) -> str:
    if idx == NC_INDEX:
        return 'NC'
    root = ROOTS[idx // N_QUALITIES]
    qual = ['maj', '7', 'm', 'dim'][idx % N_QUALITIES]
    return f'{root}{qual}'

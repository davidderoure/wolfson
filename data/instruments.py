"""
Instrument family definitions using WJD instrument codes.

WJD codes found in the solo_info.instrument column.
Add new families here to extend the system to trumpet, trombone, flute, etc.
"""

INSTRUMENT_FAMILIES = {
    "sax": ["as", "ts", "bs", "ss"],   # alto, tenor, baritone, soprano
    "trumpet":  ["tp", "cor", "cnt", "flh"],  # trumpet, cornet, flugelhorn
    "trombone": ["tb", "btb"],         # trombone, bass trombone
    "flute":    ["fl", "afl", "pfl"],  # flute, alto flute, piccolo
    "clarinet": ["cl", "bcl"],         # clarinet, bass clarinet
}

# Pitch ranges (MIDI) comfortable for each family — used to clip/validate notes
PITCH_RANGES = {
    "sax":      (44, 93),   # Ab2–A6 covers all sax voices
    "trumpet":  (52, 84),   # E3–C6
    "trombone": (40, 72),   # E2–C5
    "flute":    (60, 96),   # C4–C7
    "clarinet": (50, 91),   # D3–G6
}


def codes_for(family: str) -> list[str]:
    if family not in INSTRUMENT_FAMILIES:
        raise ValueError(f"Unknown family '{family}'. Choose from: {list(INSTRUMENT_FAMILIES)}")
    return INSTRUMENT_FAMILIES[family]


def family_for_code(code: str) -> str | None:
    for family, codes in INSTRUMENT_FAMILIES.items():
        if code in codes:
            return family
    return None

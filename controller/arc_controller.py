"""
Macro-level structural arc controller.

Tracks elapsed time and decides how the sax should respond:
- what generation parameters to use (temperature, phrase length)
- whether to generate fresh material or recall/vary a stored phrase
"""

import time
import random
from config import ARC, GENERATION_TEMPERATURE
from memory.phrase_memory import PhraseMemory


class ArcController:

    STAGES = ["sparse", "building", "peak", "recapitulation", "resolution"]

    def __init__(self, memory: PhraseMemory):
        self.memory = memory
        self._start_time = None

    def start(self):
        self._start_time = time.time()

    def elapsed(self):
        return time.time() - self._start_time if self._start_time else 0

    def stage(self):
        t = self.elapsed()
        for name in self.STAGES:
            start, end = ARC[name]
            if start <= t < end:
                return name
        return "resolution"

    def get_response_params(self, bass_phrase):
        """
        Decide how to respond to the current bass phrase.
        Returns a dict consumed by main.py to drive the generator.
        """
        stage = self.stage()
        t = self.elapsed()

        if stage == "sparse":
            return {
                "mode": "generate",
                "seed": bass_phrase,
                "n_notes": random.randint(3, 6),
                "temperature": 0.8,
            }

        elif stage == "building":
            # Mix: mostly generate, occasionally echo a recent bass phrase
            if random.random() < 0.3 and self.memory.recall_recent("bass"):
                seed = self.memory.recall_recent("bass", n=1)[0]
                return {"mode": "recall", "seed": seed, "n_notes": random.randint(6, 10), "temperature": 0.9}
            return {"mode": "generate", "seed": bass_phrase, "n_notes": random.randint(6, 10), "temperature": 0.9}

        elif stage == "peak":
            return {
                "mode": "generate",
                "seed": bass_phrase,
                "n_notes": random.randint(10, 16),
                "temperature": 1.05,   # slightly more adventurous
            }

        elif stage == "recapitulation":
            # Prefer recalling early material
            early = self.memory.recall_early("bass", n=4)
            if early:
                seed = random.choice(early)
                return {"mode": "recall", "seed": seed, "n_notes": random.randint(6, 10), "temperature": 0.85}
            return {"mode": "generate", "seed": bass_phrase, "n_notes": random.randint(6, 10), "temperature": 0.85}

        else:  # resolution
            return {
                "mode": "generate",
                "seed": bass_phrase,
                "n_notes": random.randint(2, 5),
                "temperature": 0.7,
            }

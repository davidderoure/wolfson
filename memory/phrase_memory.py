"""Stores phrases played by bass and sax for later recall and development."""

import random
from config import MAX_PHRASES_STORED


class PhraseMemory:
    """
    Circular buffer of phrases. Each entry records who played it and when.
    Provides retrieval strategies: recent, random, early (for recapitulation).
    """

    def __init__(self):
        self._phrases = []   # list of {phrase, source, index}
        self._counter = 0

    def store(self, phrase, source="bass"):
        entry = {"phrase": phrase, "source": source, "index": self._counter}
        self._counter += 1
        self._phrases.append(entry)
        if len(self._phrases) > MAX_PHRASES_STORED:
            self._phrases.pop(0)

    def recall_recent(self, source=None, n=1):
        pool = self._filter(source)
        return [e["phrase"] for e in pool[-n:]]

    def recall_random(self, source=None):
        pool = self._filter(source)
        return random.choice(pool)["phrase"] if pool else None

    def recall_early(self, source=None, n=4):
        """Return phrases from the first part of the performance — for recapitulation."""
        pool = self._filter(source)
        return [e["phrase"] for e in pool[:n]]

    def _filter(self, source):
        if source is None:
            return self._phrases
        return [e for e in self._phrases if e["source"] == source]

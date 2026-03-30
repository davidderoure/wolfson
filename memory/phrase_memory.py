"""Stores phrases played by bass and sax for later recall and development."""

import random
from collections import Counter
from config import MAX_PHRASES_STORED


class PhraseMemory:
    """
    Circular buffer of phrases. Each entry records who played it and when.
    Provides retrieval strategies: recent, random, early (for recapitulation).
    Also stores interval motifs per phrase for motivic development queries.
    """

    def __init__(self):
        self._phrases = []   # list of {phrase, source, index, motifs}
        self._counter = 0

    def store(self, phrase, source="bass", motifs=None, lyrical_motifs=None):
        entry = {
            "phrase":          phrase,
            "source":          source,
            "index":           self._counter,
            "motifs":          motifs or [],
            "lyrical_motifs":  lyrical_motifs or [],
        }
        self._counter += 1
        self._phrases.append(entry)
        if len(self._phrases) > MAX_PHRASES_STORED:
            self._phrases.pop(0)

    def recall_motifs(self, source=None, n_recent: int = 16) -> Counter:
        """
        Return a Counter of interval motifs seen in the last n_recent phrases.
        Used by the arc controller to identify recurring patterns for development.
        """
        pool = self._filter(source)[-n_recent:]
        counter = Counter()
        for entry in pool:
            for motif in entry.get("motifs", []):
                counter[motif] += 1
        return counter

    def recall_lyrical_motifs(self, source=None, n_recent: int = 16) -> Counter:
        """
        Return a Counter of *lyrical* (long-note) interval motifs seen in the
        last n_recent phrases.  These are the sustained, singable shapes that
        are worth quoting back during quieter arc stages (recap, resolution).
        """
        pool = self._filter(source)[-n_recent:]
        counter = Counter()
        for entry in pool:
            for motif in entry.get("lyrical_motifs", []):
                counter[motif] += 1
        return counter

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

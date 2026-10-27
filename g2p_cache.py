"""High-Performance Lexical G2P Front-End with LRU Caching and Fast-Path Normalizer.
Module: g2p_cache.py

Provides a 32,768-entry LRU word-phoneme cache and fast normalizer to bypass
heavy NLP/G2P passes for standard non-homograph vocabulary.
Standard library only.
"""
from __future__ import annotations

import re
import threading
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Set, Callable, Any

# Known English homographs that require contextual NLP/POS disambiguation
HOMOGRAPHS: Set[str] = frozenset({
    "read", "lead", "record", "tear", "wind", "live", "wound", "bow", "bass",
    "close", "contract", "desert", "minute", "object", "produce", "rebel",
    "refuse", "row", "subject", "use"
})

_WORD_CLEAN_RE = re.compile(r"[^\w']+")
_WHITESPACE_RE = re.compile(r"\s+")


class FastNormalizer:
    """Zero-overhead text cleaner and ASCII normalizer for speech front-end."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Collapse whitespace and normalize common typographic unicode punctuation."""
        if not text:
            return ""
        # Fast replace common typographic quotes and dashes
        t = text.replace("“", '"').replace("”", '"')
        t = t.replace("‘", "'").replace("’", "'")
        t = t.replace("—", " - ").replace("–", " - ")
        return _WHITESPACE_RE.sub(" ", t).strip()

    @staticmethod
    def clean_word(word: str) -> str:
        """Extract lexical alphabetic/apostrophe core in lowercase."""
        return _WORD_CLEAN_RE.sub("", word).lower()


class LexicalPhonemeCache:
    """Thread-safe LRU cache mapping normalized words to phoneme token sequences."""

    def __init__(self, capacity: int = 32768):
        self.capacity = capacity
        self.hits = 0
        self.misses = 0
        self._cache: OrderedDict[str, Tuple[str, Tuple[int, ...]]] = OrderedDict()
        self._backend_g2p: Optional[Callable[..., Any]] = None
        self._lock = threading.Lock()

    def set_backend_g2p(self, g2p_callable: Callable[..., Any]) -> None:
        """Bind underlying Misaki, eSpeak, or Kokoro G2P callable."""
        with self._lock:
            self._backend_g2p = g2p_callable

    def contains_homograph(self, text: str) -> bool:
        """Fast check if text contains any ambiguous homographs."""
        words = _WORD_CLEAN_RE.sub(" ", text).lower().split()
        return any(w in HOMOGRAPHS for w in words)

    def _phonemize_word_uncached(self, clean_word: str) -> Tuple[str, Tuple[int, ...]]:
        """Executed only on cache miss."""
        if self._backend_g2p is not None:
            try:
                res = self._backend_g2p(clean_word)
                if isinstance(res, tuple) and len(res) == 2:
                    phonemes, ids = res
                    return str(phonemes), tuple(int(x) for x in ids)
            except Exception:
                pass
        # Fallback deterministic pseudo-phonemes / token IDs for testing or unhooked state
        phonemes = f"/{clean_word}/"
        ids = tuple((ord(c) % 128) + 1 for c in clean_word)
        return phonemes, ids

    def get_word_phonemes(self, clean_word: str) -> Tuple[str, Tuple[int, ...]]:
        """Retrieve phonemes and token IDs for a single cleaned word from cache or backend."""
        if not clean_word:
            return "", ()

        with self._lock:
            if clean_word in self._cache:
                self._cache.move_to_end(clean_word)
                self.hits += 1
                return self._cache[clean_word]

            self.misses += 1
            result = self._phonemize_word_uncached(clean_word)
            self._cache[clean_word] = result
            if len(self._cache) > self.capacity:
                self._cache.popitem(last=False)
            return result

    def process_clause(
        self,
        clause_text: str,
        source_indices: Tuple[int, ...]
    ) -> Tuple[List[int], List[int]]:
        """Converts clause text to phoneme token IDs and maintains source word indices.

        Returns:
            token_ids: Model input token IDs
            token_to_source_word: Index map of each token to its source display word
        """
        words = clause_text.split()
        token_ids: List[int] = []
        token_to_word: List[int] = []

        # If homographs present or contextual method available, invoke contextual parser
        if self.contains_homograph(clause_text):
            return self._contextual_process(clause_text, source_indices)

        # Fast-path: Resolve word by word via LRU cache
        for i, word in enumerate(words):
            clean_word = FastNormalizer.clean_word(word)
            source_idx = source_indices[i] if i < len(source_indices) else -1

            if not clean_word:
                continue

            _, ids = self.get_word_phonemes(clean_word)
            for tid in ids:
                token_ids.append(tid)
                token_to_word.append(source_idx)

        return token_ids, token_to_word

    def _contextual_process(
        self,
        clause_text: str,
        source_indices: Tuple[int, ...]
    ) -> Tuple[List[int], List[int]]:
        """Runs full contextual G2P and associates tokens with word boundaries."""
        with self._lock:
            backend = self._backend_g2p

        if backend is not None and hasattr(backend, "contextual"):
            try:
                return backend.contextual(clause_text, source_indices)
            except Exception:
                pass

        # If no contextual backend method, parse word by word with provenance
        words = clause_text.split()
        token_ids: List[int] = []
        token_to_word: List[int] = []
        for i, word in enumerate(words):
            clean_word = FastNormalizer.clean_word(word)
            source_idx = source_indices[i] if i < len(source_indices) else -1
            if not clean_word:
                continue
            _, ids = self.get_word_phonemes(clean_word)
            for tid in ids:
                token_ids.append(tid)
                token_to_word.append(source_idx)
        return token_ids, token_to_word

    def get_stats(self) -> Dict[str, Any]:
        """Return diagnostic metrics on cache usage."""
        with self._lock:
            total = self.hits + self.misses
            hit_rate = (self.hits / total) if total > 0 else 0.0
            return {
                "capacity": self.capacity,
                "size": len(self._cache),
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": round(hit_rate, 4),
            }

    def clear(self) -> None:
        """Reset the cache and counters."""
        with self._lock:
            self._cache.clear()
            self.hits = 0
            self.misses = 0

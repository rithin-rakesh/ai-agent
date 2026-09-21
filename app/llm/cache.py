"""Semantic Cache Abstraction.

Implements an in-memory SHA256 content-hashed cache for embedding vectors
and LLM evaluations to prevent redundant API calls for unchanged text.
"""

import hashlib
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class SemanticCache:
    """In-memory cache using content hashing for embeddings and evaluations."""

    def __init__(self) -> None:
        self._embedding_cache: Dict[str, List[float]] = {}
        self._reasoning_cache: Dict[str, Any] = {}
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _hash_text(text: str) -> str:
        """Compute SHA-256 hash of normalized text string."""
        normalized = " ".join(text.strip().split())
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    def get_embedding(
        self,
        text: str,
        model: str,
        input_type: str,
    ) -> Optional[List[float]]:
        """Retrieve cached embedding if present."""
        key = f"emb:{model}:{input_type}:{self._hash_text(text)}"
        if key in self._embedding_cache:
            self._hits += 1
            return self._embedding_cache[key]
        self._misses += 1
        return None

    def set_embedding(
        self,
        text: str,
        model: str,
        input_type: str,
        embedding: List[float],
    ) -> None:
        """Store embedding vector in cache."""
        key = f"emb:{model}:{input_type}:{self._hash_text(text)}"
        self._embedding_cache[key] = embedding

    def get_reasoning(self, key_text: str, model: str) -> Optional[Any]:
        """Retrieve cached reasoning result."""
        key = f"rsn:{model}:{self._hash_text(key_text)}"
        if key in self._reasoning_cache:
            self._hits += 1
            return self._reasoning_cache[key]
        self._misses += 1
        return None

    def set_reasoning(self, key_text: str, model: str, data: Any) -> None:
        """Store reasoning evaluation in cache."""
        key = f"rsn:{model}:{self._hash_text(key_text)}"
        self._reasoning_cache[key] = data

    def clear(self) -> None:
        """Clear all cached entries."""
        self._embedding_cache.clear()
        self._reasoning_cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def stats(self) -> Dict[str, int]:
        """Return cache performance statistics."""
        return {
            "embedding_entries": len(self._embedding_cache),
            "reasoning_entries": len(self._reasoning_cache),
            "hits": self._hits,
            "misses": self._misses,
        }

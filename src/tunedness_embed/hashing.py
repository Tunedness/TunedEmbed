"""Deterministic fake embedder for tests: no network, no model files."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence

import numpy as np

from .base import Matrix, check_texts, l2_normalize

_TOKEN_RE = re.compile(r"\w+")


class HashEmbedder:
    """Feature-hashing embedder.

    Each lower-cased word token is hashed (SHA-256) to a signed bucket of the vector;
    rows are then L2-normalized. Texts sharing words get a high cosine similarity,
    which is enough to exercise downstream code deterministically. It carries no
    semantic knowledge and must not be used for real measurements.
    """

    def __init__(self, dimension: int = 64) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._dimension = dimension

    @property
    def model_id(self) -> str:
        return f"hash-{self._dimension}"

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> Matrix:
        items = check_texts(texts)
        out = np.zeros((len(items), self._dimension), dtype=np.float32)
        for row, text in enumerate(items):
            # An empty text still gets a stable, non-zero vector.
            tokens = _TOKEN_RE.findall(text.lower()) or [""]
            for token in tokens:
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                bucket = int.from_bytes(digest[:8], "big") % self._dimension
                sign = 1.0 if digest[8] & 1 else -1.0
                out[row, bucket] += sign
        # Opposite-sign collisions can cancel to zero; fall back to a fixed unit vector.
        zero_rows = ~out.any(axis=1)
        out[zero_rows, 0] = 1.0
        return l2_normalize(out)

"""Embedder protocol, shared helpers and error types."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

Matrix = NDArray[np.float32]


class TunedEmbedError(Exception):
    """Base class for all errors raised by this package."""


class ModelNotAvailableError(TunedEmbedError):
    """Model files are not cached and could not be obtained (offline or download failure)."""


class ModelIntegrityError(TunedEmbedError):
    """A downloaded model file did not match its pinned size or SHA-256."""


@runtime_checkable
class Embedder(Protocol):
    """Turns texts into L2-normalized float32 vectors."""

    @property
    def model_id(self) -> str:
        """Stable identifier of the embedding model (recorded in reports)."""
        ...

    @property
    def dimension(self) -> int:
        """Length of each embedding vector."""
        ...

    def embed(self, texts: Sequence[str]) -> Matrix:
        """Return a ``(len(texts), dimension)`` float32 matrix with unit-length rows."""
        ...


def check_texts(texts: Sequence[str]) -> list[str]:
    """Validate the ``embed`` input and return it as a list."""
    if isinstance(texts, str):
        raise TypeError("texts must be a sequence of strings, not a single string")
    items = list(texts)
    for item in items:
        if not isinstance(item, str):
            raise TypeError(f"texts must contain only strings, got {type(item).__name__}")
    return items


def l2_normalize(matrix: Matrix) -> Matrix:
    """Scale every row to unit length; all-zero rows are left as zeros."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0).astype(np.float32)
    return (matrix / safe).astype(np.float32, copy=False)

"""tunedness-embed: small local ONNX text-embedding library."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import (
    Embedder,
    ModelIntegrityError,
    ModelNotAvailableError,
    TunedEmbedError,
)
from .hashing import HashEmbedder

if TYPE_CHECKING:
    from .onnx import OnnxEmbedder

__all__ = [
    "Embedder",
    "HashEmbedder",
    "ModelIntegrityError",
    "ModelNotAvailableError",
    "OnnxEmbedder",
    "TunedEmbedError",
]

__version__ = "0.1.0"


def __getattr__(name: str) -> Any:
    # Lazy import: onnxruntime is only loaded when OnnxEmbedder is actually used.
    if name == "OnnxEmbedder":
        from .onnx import OnnxEmbedder

        return OnnxEmbedder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

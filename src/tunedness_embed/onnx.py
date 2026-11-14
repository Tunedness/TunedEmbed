"""ONNX Runtime embedder (CPU, mean pooling)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from .base import Matrix, check_texts, l2_normalize
from .model_store import ensure_model, get_model_spec


class OnnxEmbedder:
    """Sentence embedder backed by a local ONNX model.

    The model files are downloaded on first use into ``cache_dir`` (default
    ``~/.cache/tunedness-embed/``) and verified against SHA-256 values pinned in code.
    With ``offline=True`` (or ``TUNEDNESS_EMBED_OFFLINE=1``) nothing is downloaded and a
    missing model raises ``ModelNotAvailableError``.
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        cache_dir: Path | None = None,
        offline: bool = False,
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        spec = get_model_spec(model)
        directory = ensure_model(spec, cache_dir=cache_dir, offline=offline)

        self._spec = spec
        self._batch_size = batch_size
        self._tokenizer = Tokenizer.from_file(str(directory / spec.tokenizer_file))
        self._tokenizer.enable_truncation(max_length=spec.max_length)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")  # noqa: S106
        self._session = ort.InferenceSession(
            str(directory / spec.model_file), providers=["CPUExecutionProvider"]
        )
        self._input_names = {item.name for item in self._session.get_inputs()}

    @property
    def model_id(self) -> str:
        return self._spec.model_id

    @property
    def dimension(self) -> int:
        return self._spec.dimension

    def embed(self, texts: Sequence[str]) -> Matrix:
        items = check_texts(texts)
        if not items:
            return np.zeros((0, self.dimension), dtype=np.float32)
        batches = [
            self._embed_batch(items[start : start + self._batch_size])
            for start in range(0, len(items), self._batch_size)
        ]
        return l2_normalize(np.concatenate(batches, axis=0))

    def _embed_batch(self, batch: list[str]) -> Matrix:
        encodings = self._tokenizer.encode_batch(batch)
        input_ids = np.array([e.ids for e in encodings], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        candidates = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": np.zeros_like(input_ids),
        }
        feed = {name: value for name, value in candidates.items() if name in self._input_names}
        hidden = np.asarray(self._session.run(None, feed)[0], dtype=np.float32)
        # Mean pooling over real (non-padding) tokens.
        mask = attention_mask[:, :, np.newaxis].astype(np.float32)
        summed = (hidden * mask).sum(axis=1)
        counts = np.clip(mask.sum(axis=1), 1e-9, None)
        return (summed / counts).astype(np.float32, copy=False)

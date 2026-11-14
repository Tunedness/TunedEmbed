"""Pinned model files: download, SHA-256 verification and caching."""

from __future__ import annotations

import contextlib
import hashlib
import http.client
import os
import tempfile
import urllib.request
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .base import ModelIntegrityError, ModelNotAvailableError

OFFLINE_ENV_VAR = "TUNEDNESS_EMBED_OFFLINE"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})
_CHUNK_SIZE = 1024 * 1024
_HTTP_TIMEOUT_SECONDS = 60

# Returns the response body for a URL as an iterable of byte chunks.
Fetcher = Callable[[str], Iterator[bytes]]


@dataclass(frozen=True)
class ModelFile:
    """One downloadable file. ``sha256`` and ``size`` are pinned in code."""

    name: str
    url: str
    sha256: str
    size: int


@dataclass(frozen=True)
class ModelSpec:
    """A model made of an ONNX graph and a tokenizer definition."""

    model_id: str
    dimension: int
    max_length: int
    model_file: str
    tokenizer_file: str
    files: tuple[ModelFile, ...]
    license: str
    source: str


# Pinned to an immutable Hugging Face commit; hashes were computed from the real files.
_MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
_MINILM_BASE = (
    f"https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/{_MINILM_REVISION}"
)

MODELS: dict[str, ModelSpec] = {
    "all-MiniLM-L6-v2": ModelSpec(
        model_id="all-MiniLM-L6-v2",
        dimension=384,
        max_length=256,
        model_file="model.onnx",
        tokenizer_file="tokenizer.json",
        files=(
            ModelFile(
                name="model.onnx",
                url=f"{_MINILM_BASE}/onnx/model.onnx",
                sha256="6fd5d72fe4589f189f8ebc006442dbb529bb7ce38f8082112682524616046452",
                size=90405214,
            ),
            ModelFile(
                name="tokenizer.json",
                url=f"{_MINILM_BASE}/tokenizer.json",
                sha256="be50c3628f2bf5bb5e3a7f17b1f74611b2561a3a27eeab05e5aa30f411572037",
                size=466247,
            ),
        ),
        license="Apache-2.0",
        source="https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2",
    ),
}


def get_model_spec(model: str) -> ModelSpec:
    """Look a model up by name; unknown names are rejected (they also name cache dirs)."""
    try:
        return MODELS[model]
    except KeyError:
        known = ", ".join(sorted(MODELS))
        raise ValueError(f"unknown model {model!r}; supported models: {known}") from None


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "tunedness-embed"


def is_offline(offline: bool = False) -> bool:
    """True if the caller asked for offline mode or ``TUNEDNESS_EMBED_OFFLINE`` is set."""
    if offline:
        return True
    return os.environ.get(OFFLINE_ENV_VAR, "").strip().lower() in _TRUE_VALUES


def http_fetch(url: str) -> Iterator[bytes]:
    """Default fetcher: streams an HTTPS URL."""
    if not url.startswith("https://"):
        raise ValueError(f"refusing to download from a non-HTTPS URL: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "tunedness-embed"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_SECONDS) as response:  # noqa: S310
        while chunk := response.read(_CHUNK_SIZE):
            yield chunk


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _is_valid(path: Path, file: ModelFile) -> bool:
    return path.is_file() and path.stat().st_size == file.size and file_sha256(path) == file.sha256


def _download(file: ModelFile, target: Path, fetch: Fetcher) -> None:
    """Download to a temp file, verify, then atomically move into place."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{file.name}.", suffix=".part")
    tmp_path = Path(tmp_name)
    try:
        digest = hashlib.sha256()
        total = 0
        with os.fdopen(fd, "wb") as handle:
            for chunk in fetch(file.url):
                total += len(chunk)
                if total > file.size:
                    raise ModelIntegrityError(
                        f"{file.name}: download exceeds the pinned size of {file.size} bytes"
                    )
                digest.update(chunk)
                handle.write(chunk)
        if total < file.size:
            raise ModelNotAvailableError(
                f"{file.name}: incomplete download ({total} of {file.size} bytes); "
                "the connection was probably interrupted. The partial file was discarded."
            )
        if total != file.size or digest.hexdigest() != file.sha256:
            raise ModelIntegrityError(
                f"{file.name}: downloaded content does not match the pinned SHA-256 "
                f"(expected {file.sha256}, got {digest.hexdigest()}; {total} bytes). "
                "The file was discarded."
            )
        os.replace(tmp_path, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp_path.unlink()
        raise


def ensure_model(
    spec: ModelSpec,
    cache_dir: Path | None = None,
    offline: bool = False,
    fetch: Fetcher | None = None,
) -> Path:
    """Make sure every file of ``spec`` is cached and verified; return the model directory.

    Cached files are re-verified on every call. A cached file that fails verification is
    deleted and treated as missing. A freshly downloaded file that fails verification is
    deleted and raises :class:`ModelIntegrityError`.
    """
    directory = (cache_dir if cache_dir is not None else default_cache_dir()) / spec.model_id
    is_offline_mode = is_offline(offline)
    fetcher = fetch if fetch is not None else http_fetch

    for file in spec.files:
        path = directory / file.name
        if _is_valid(path, file):
            continue
        if path.exists():
            path.unlink()
        if is_offline_mode:
            raise ModelNotAvailableError(
                f"Model {spec.model_id!r} is not available offline: {path} is missing or "
                f"corrupt. Offline mode is on (offline=True or {OFFLINE_ENV_VAR}=1). "
                f"Run once without offline mode to download it into {directory.parent}, "
                "or copy the pre-downloaded files there."
            )
        try:
            _download(file, path, fetcher)
        except (OSError, http.client.HTTPException, ValueError) as error:
            raise ModelNotAvailableError(
                f"Could not download {file.name} for model {spec.model_id!r} "
                f"from {file.url}: {error}"
            ) from error
    return directory

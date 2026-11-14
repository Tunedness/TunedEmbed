import hashlib
import http.client
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path

import pytest

from tunedness_embed import ModelIntegrityError, ModelNotAvailableError
from tunedness_embed.model_store import (
    MODELS,
    OFFLINE_ENV_VAR,
    ModelFile,
    ModelSpec,
    ensure_model,
    get_model_spec,
    http_fetch,
    is_offline,
)

PAYLOAD = b"fake-onnx-bytes" * 1000
URL = "https://example.invalid/model.bin"


def make_spec(payload: bytes = PAYLOAD) -> ModelSpec:
    file = ModelFile(
        name="model.bin",
        url=URL,
        sha256=hashlib.sha256(payload).hexdigest(),
        size=len(payload),
    )
    return ModelSpec(
        model_id="fake-model",
        dimension=4,
        max_length=8,
        model_file="model.bin",
        tokenizer_file="model.bin",
        files=(file,),
        license="Apache-2.0",
        source="https://example.invalid",
    )


class FakeFetcher:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.calls: list[str] = []

    def __call__(self, url: str) -> Iterator[bytes]:
        self.calls.append(url)
        for start in range(0, len(self.payload), 700):
            yield self.payload[start : start + 700]


def leftovers(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.iterdir())


def test_download_verifies_and_caches(tmp_path: Path) -> None:
    fetcher = FakeFetcher(PAYLOAD)
    directory = ensure_model(make_spec(), cache_dir=tmp_path, fetch=fetcher)
    assert directory == tmp_path / "fake-model"
    assert (directory / "model.bin").read_bytes() == PAYLOAD
    assert leftovers(directory) == ["model.bin"]  # no temp files left behind

    ensure_model(make_spec(), cache_dir=tmp_path, fetch=fetcher)
    assert fetcher.calls == [URL]  # second call served from cache


def test_corrupt_download_raises_integrity_error_and_is_deleted(tmp_path: Path) -> None:
    tampered = b"X" + PAYLOAD[1:]  # same size, different content
    with pytest.raises(ModelIntegrityError):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=FakeFetcher(tampered))
    assert leftovers(tmp_path / "fake-model") == []


def test_truncated_download_is_reported_as_incomplete_not_as_tampering(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError, match="incomplete download"):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=FakeFetcher(PAYLOAD[:-5]))
    assert leftovers(tmp_path / "fake-model") == []


def test_oversized_download_is_aborted(tmp_path: Path) -> None:
    with pytest.raises(ModelIntegrityError):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=FakeFetcher(PAYLOAD + b"extra"))
    assert leftovers(tmp_path / "fake-model") == []


def test_dropped_connection_raises_not_available_and_leaves_no_files(tmp_path: Path) -> None:
    def boom(url: str) -> Iterator[bytes]:
        yield PAYLOAD[:10]
        raise ConnectionResetError("connection dropped")

    with pytest.raises(ModelNotAvailableError):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=boom)
    assert leftovers(tmp_path / "fake-model") == []


def test_incomplete_read_is_wrapped_in_package_error(tmp_path: Path) -> None:
    def boom(url: str) -> Iterator[bytes]:
        yield PAYLOAD[:10]
        raise http.client.IncompleteRead(PAYLOAD[:10], 100)

    with pytest.raises(ModelNotAvailableError):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=boom)
    assert leftovers(tmp_path / "fake-model") == []


def test_non_https_url_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        list(http_fetch("http://example.invalid/model.bin"))

    valid = make_spec().files[0]
    insecure_file = replace(valid, url="http://example.invalid/model.bin")
    spec = replace(make_spec(), files=(insecure_file,))
    with pytest.raises(ModelNotAvailableError, match="non-HTTPS"):
        ensure_model(spec, cache_dir=tmp_path)  # default fetcher
    assert leftovers(tmp_path / "fake-model") == []


def test_offline_with_missing_model_raises_not_available(tmp_path: Path) -> None:
    fetcher = FakeFetcher(PAYLOAD)
    with pytest.raises(ModelNotAvailableError, match="offline"):
        ensure_model(make_spec(), cache_dir=tmp_path, offline=True, fetch=fetcher)
    assert fetcher.calls == []


def test_offline_env_var_is_honoured(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(OFFLINE_ENV_VAR, "1")
    assert is_offline() is True
    fetcher = FakeFetcher(PAYLOAD)
    with pytest.raises(ModelNotAvailableError):
        ensure_model(make_spec(), cache_dir=tmp_path, fetch=fetcher)
    assert fetcher.calls == []

    monkeypatch.setenv(OFFLINE_ENV_VAR, "0")
    assert is_offline() is False


def test_offline_uses_valid_cached_files(tmp_path: Path) -> None:
    ensure_model(make_spec(), cache_dir=tmp_path, fetch=FakeFetcher(PAYLOAD))
    directory = ensure_model(make_spec(), cache_dir=tmp_path, offline=True)
    assert (directory / "model.bin").read_bytes() == PAYLOAD


def test_corrupt_cache_is_deleted_offline_and_redownloaded_online(tmp_path: Path) -> None:
    directory = ensure_model(make_spec(), cache_dir=tmp_path, fetch=FakeFetcher(PAYLOAD))
    (directory / "model.bin").write_bytes(b"garbage")

    with pytest.raises(ModelNotAvailableError):
        ensure_model(make_spec(), cache_dir=tmp_path, offline=True)
    assert not (directory / "model.bin").exists()

    (directory / "model.bin").write_bytes(b"garbage")
    fetcher = FakeFetcher(PAYLOAD)
    ensure_model(make_spec(), cache_dir=tmp_path, fetch=fetcher)
    assert (directory / "model.bin").read_bytes() == PAYLOAD
    assert fetcher.calls == [URL]


def test_unknown_model_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown model"):
        get_model_spec("../../etc/passwd")


def test_pinned_registry_is_well_formed() -> None:
    assert "all-MiniLM-L6-v2" in MODELS
    for spec in MODELS.values():
        names = {file.name for file in spec.files}
        assert {spec.model_file, spec.tokenizer_file} <= names
        for file in spec.files:
            assert file.url.startswith("https://")
            assert len(file.sha256) == 64
            int(file.sha256, 16)
            assert file.size > 0

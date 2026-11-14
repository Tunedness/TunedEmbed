from pathlib import Path

import numpy as np
import pytest

from tunedness_embed import Embedder, ModelNotAvailableError, OnnxEmbedder


def test_offline_without_cached_model_raises(tmp_path: Path) -> None:
    with pytest.raises(ModelNotAvailableError):
        OnnxEmbedder(cache_dir=tmp_path, offline=True)


def test_offline_env_without_cached_model_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TUNEDNESS_EMBED_OFFLINE", "1")
    with pytest.raises(ModelNotAvailableError):
        OnnxEmbedder(cache_dir=tmp_path)


def test_unknown_model_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        OnnxEmbedder(model="no-such-model", cache_dir=tmp_path)


def test_invalid_batch_size(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="batch_size"):
        OnnxEmbedder(cache_dir=tmp_path, batch_size=0)


@pytest.fixture(scope="module")
def cache_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("model-cache")


@pytest.fixture(scope="module")
def embedder(cache_dir: Path) -> OnnxEmbedder:
    return OnnxEmbedder(cache_dir=cache_dir)


@pytest.mark.network
def test_downloads_model_and_embeds(embedder: OnnxEmbedder) -> None:
    assert isinstance(embedder, Embedder)
    assert embedder.model_id == "all-MiniLM-L6-v2"
    assert embedder.dimension == 384

    vectors = embedder.embed(["hello world", "", "a much longer sentence " * 100])
    assert vectors.shape == (3, 384)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)


@pytest.mark.network
def test_similar_sentences_score_higher_than_unrelated(embedder: OnnxEmbedder) -> None:
    anchor, similar, unrelated = embedder.embed(
        [
            "A man is playing a guitar on stage.",
            "Someone is performing guitar music in front of an audience.",
            "The central bank raised interest rates by half a point.",
        ]
    )
    assert float(np.dot(anchor, similar)) > float(np.dot(anchor, unrelated))


@pytest.mark.network
def test_batching_and_padding_do_not_change_results(embedder: OnnxEmbedder) -> None:
    texts = ["short", "a considerably longer text with many more tokens in it", "mid length"]
    together = embedder.embed(texts)
    separately = np.vstack([embedder.embed([t]) for t in texts])
    np.testing.assert_allclose(together, separately, atol=1e-4)
    assert embedder.embed([]).shape == (0, 384)


@pytest.mark.network
def test_multi_batch_path_preserves_order(embedder: OnnxEmbedder, cache_dir: Path) -> None:
    texts = [
        "one",
        "two words",
        "a slightly longer third text",
        "fourth",
        "the fifth and final text in this list",
    ]
    small_batches = OnnxEmbedder(cache_dir=cache_dir, batch_size=2)  # 3 batches: 2 + 2 + 1
    batched = small_batches.embed(texts)
    singles = np.vstack([embedder.embed([t]) for t in texts])
    assert batched.shape == (5, 384)
    np.testing.assert_allclose(batched, singles, atol=1e-4)


@pytest.mark.network
def test_cached_model_reopens_offline(embedder: OnnxEmbedder, cache_dir: Path) -> None:
    offline = OnnxEmbedder(cache_dir=cache_dir, offline=True)
    np.testing.assert_allclose(offline.embed(["x"]), embedder.embed(["x"]), atol=1e-6)

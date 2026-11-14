import numpy as np
import pytest

from tunedness_embed import Embedder, HashEmbedder


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def test_shape_dtype_and_unit_norm() -> None:
    embedder = HashEmbedder(dimension=32)
    vectors = embedder.embed(["alpha beta", "gamma", ""])
    assert vectors.shape == (3, 32)
    assert vectors.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)


def test_deterministic_across_instances() -> None:
    first = HashEmbedder().embed(["the quick brown fox"])
    second = HashEmbedder().embed(["the quick brown fox"])
    np.testing.assert_array_equal(first, second)


def test_shared_words_are_more_similar_than_unrelated_text() -> None:
    embedder = HashEmbedder(dimension=256)
    base, similar, unrelated = embedder.embed(
        [
            "the cat sat on the mat",
            "the cat sat on a mat",
            "quarterly revenue exceeded forecasts",
        ]
    )
    assert cosine(base, similar) > cosine(base, unrelated)


def test_case_and_punctuation_do_not_matter() -> None:
    embedder = HashEmbedder()
    a, b = embedder.embed(["Hello, World!", "hello world"])
    np.testing.assert_array_equal(a, b)


def test_empty_list_returns_empty_matrix() -> None:
    vectors = HashEmbedder(dimension=16).embed([])
    assert vectors.shape == (0, 16)
    assert vectors.dtype == np.float32


def test_empty_text_still_unit_length() -> None:
    vector = HashEmbedder().embed([""])[0]
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_rejects_bare_string_and_non_strings() -> None:
    embedder = HashEmbedder()
    with pytest.raises(TypeError):
        embedder.embed("not a sequence")
    with pytest.raises(TypeError):
        embedder.embed(["ok", 3])  # type: ignore[list-item]


def test_invalid_dimension() -> None:
    with pytest.raises(ValueError):
        HashEmbedder(dimension=0)


def test_satisfies_embedder_protocol() -> None:
    embedder = HashEmbedder(dimension=8)
    assert isinstance(embedder, Embedder)
    assert embedder.model_id == "hash-8"
    assert embedder.dimension == 8

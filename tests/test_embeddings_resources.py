"""Real MS2DeepScore model tests using the tiny repository resources."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ms2query_db.embeddings import MS2DeepScoreEmbedder, validated_embeddings
from ms2query_db.errors import RejectedRecord
from ms2query_db.io import _mgf_records, load_mgf_batch
from ms2query_db.util import file_sha256

RESOURCES = Path(__file__).parent / "resources"
TEST_MGF = RESOURCES / "test_spectra.mgf"
TEST_MODEL_PT = RESOURCES / "testmodel_v1.pt"
TEST_MODEL_ONNX = RESOURCES / "testmodel_v1.onnx"


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.fail(f"Required test resource is missing: {path}")
    return path


def _collection():
    records = list(_mgf_records(_require(TEST_MGF)))
    loaded = load_mgf_batch(records)
    assert loaded.rejected == {}
    assert loaded.collection is not None
    return loaded.collection


def _matrix(values):
    rejected = [value for value in values if isinstance(value, RejectedRecord)]
    assert not rejected, [f"{value.code}: {value}" for value in rejected]
    return np.stack(values)


def test_real_pytorch_model_produces_valid_normalized_embeddings():
    """Load the actual toy .pt model through the public MS2Query-DB adapter."""
    pytest.importorskip("ms2deepscore")
    path = _require(TEST_MODEL_PT)
    embedder = MS2DeepScoreEmbedder.from_file(
        path,
        allow_legacy=True,
        batch_size=16,
    )

    matrix = _matrix(validated_embeddings(embedder, _collection()))

    assert matrix.shape == (len(_collection()), embedder.dimension)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(
        np.linalg.norm(matrix, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )
    assert embedder.model_id == f"sha256:{file_sha256(path)}"


def test_real_onnx_model_produces_valid_normalized_embeddings():
    """Exercise ONNX loading/inference through the same public adapter."""
    pytest.importorskip("ms2deepscore")
    pytest.importorskip("onnxruntime")
    path = _require(TEST_MODEL_ONNX)
    embedder = MS2DeepScoreEmbedder.from_file(path, batch_size=16)

    collection = _collection()
    matrix = _matrix(validated_embeddings(embedder, collection))

    assert matrix.shape == (len(collection), embedder.dimension)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    np.testing.assert_allclose(
        np.linalg.norm(matrix, axis=1),
        1.0,
        rtol=1e-5,
        atol=1e-6,
    )
    assert embedder.model_id == f"sha256:{file_sha256(path)}"


def test_pt_and_onnx_versions_of_toy_model_agree_on_embeddings():
    """The converted ONNX toy model should preserve the PyTorch embedding space."""
    pytest.importorskip("ms2deepscore")
    pytest.importorskip("onnxruntime")

    pt = MS2DeepScoreEmbedder.from_file(
        _require(TEST_MODEL_PT),
        allow_legacy=True,
        batch_size=16,
    )
    onnx = MS2DeepScoreEmbedder.from_file(
        _require(TEST_MODEL_ONNX),
        batch_size=16,
    )
    assert pt.dimension == onnx.dimension

    collection = _collection()
    pt_matrix = _matrix(validated_embeddings(pt, collection))
    onnx_matrix = _matrix(validated_embeddings(onnx, collection))

    # Both matrices are L2-normalized by validated_embeddings(). Comparing the
    # per-row cosine is more robust to tiny backend-specific floating-point
    # differences than requiring bit-identical vectors.
    row_cosines = np.sum(pt_matrix * onnx_matrix, axis=1)
    assert np.min(row_cosines) > 0.999

"""Resource-backed tests for the real matchms indexed-search adapter."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ms2query_db.config import ScoringSpec
from ms2query_db.io import _mgf_records, load_mgf_batch
from ms2query_db.search.native import (
    build_index,
    load_index,
    make_scorer,
    save_index,
    search_arrays,
)

RESOURCES = Path(__file__).parents[1] / "resources"
TEST_MGF = RESOURCES / "test_spectra.mgf"


def _collection():
    if not TEST_MGF.is_file():
        pytest.fail(f"Required test resource is missing: {TEST_MGF}")
    records = list(_mgf_records(TEST_MGF))
    loaded = load_mgf_batch(records)
    assert loaded.rejected == {}
    assert loaded.collection is not None
    return loaded.collection


@pytest.mark.parametrize("method", ["cosine", "entropy_hybrid"])
def test_real_native_index_self_search_has_unit_diagonal(method):
    """Build and query the real matchms index on the resource spectra."""
    spectra = _collection()
    scorer = make_scorer(method, ScoringSpec())
    index = build_index(scorer, spectra)

    scores, matches = search_arrays(
        scorer,
        spectra,
        index,
        expected_columns=len(spectra),
        n_jobs=1,
    )

    assert scores.shape == (len(spectra), len(spectra))
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(
        np.diag(scores),
        1.0,
        rtol=1e-6,
        atol=1e-6,
    )
    if matches is not None:
        assert matches.shape == scores.shape
        assert np.all(np.diag(matches) > 0)


def test_real_entropy_index_persistence_preserves_search_results(tmp_path):
    """A saved/reloaded native index must score identically to the original."""
    spectra = _collection()
    scorer = make_scorer("entropy_hybrid", ScoringSpec())
    index = build_index(scorer, spectra)

    before, _ = search_arrays(
        scorer,
        spectra,
        index,
        expected_columns=len(spectra),
        n_jobs=1,
    )

    path = tmp_path / "entropy_index.npz"
    save_index(scorer, index, path)
    reloaded = load_index(scorer, path)

    after, _ = search_arrays(
        scorer,
        spectra,
        reloaded,
        expected_columns=len(spectra),
        n_jobs=1,
    )

    np.testing.assert_allclose(after, before, rtol=1e-12, atol=1e-12)

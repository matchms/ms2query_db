from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def make_spectrum():
    """Create small deterministic matchms spectra for unit tests."""
    matchms = pytest.importorskip("matchms")
    Spectrum = matchms.Spectrum

    def factory(
        *,
        precursor_mz: object = 250.0,
        ionmode: str = "positive",
        smiles: str | None = None,
        mz=(50.0, 100.0, 150.0),
        intensities=(1.0, 0.5, 0.25),
        **metadata,
    ):
        values = {
            "precursor_mz": precursor_mz,
            "ionmode": ionmode,
            **metadata,
        }
        if smiles is not None:
            values["smiles"] = smiles
        return Spectrum(
            mz=np.asarray(mz, dtype=np.float64),
            intensities=np.asarray(intensities, dtype=np.float64),
            metadata=values,
            metadata_harmonization=False,
        )

    return factory


@pytest.fixture
def query_config_factory():
    """Minimal QueryConfig-like object for QueryEngine unit tests."""

    def factory(**overrides):
        values = {
            "search_methods": ("ms2deepscore", "entropy_hybrid"),
            "top_n": 20,
            "additional_top_n_scores": None,
            "batch_size": 32,
            "classical_query_batch": 32,
            "classical_n_jobs": 1,
            "ef_search": 200,
            "exact_embedding_search": False,
            "ionmode_matching": "same_mode",
            "annotation_policy": "ignore",
            "compute_chemical_similarity": False,
            "chemical_metric": "minmax",
            "aggregation_method": None,
            "aggregation_weight_source": "rf_quality",
            "aggregation_weight_power": 2.0,
            "aggregation_selection": "top_n",
            "aggregation_top_n": 20,
            "aggregation_score_delta": 0.2,
            "chemical_search_pool": 500,
            "exact_chemical_search": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    return factory


@pytest.fixture
def build_config_factory():
    """Minimal BuildConfig-like object for builder/database unit tests."""

    def factory(**overrides):
        fingerprints = SimpleNamespace(
            size=16,
            projection_dim=4,
        )
        values = {
            "schema": "test_library",
            "ionmode": "positive",
            "batch_size": 100,
            "fingerprints": fingerprints,
            "query_processing": SimpleNamespace(),
            "scoring": SimpleNamespace(),
            "classical_methods": ("entropy_hybrid",),
            "index_shard_spectra": 20_000,
            "index_shard_peaks": 1_000_000,
            "mz_precision": 1e-6,
            "recursive": False,
            "pattern": "*.mgf",
            "smiles_column": "smiles",
            "inchi_column": "inchi",
            "inchikey_column": "inchikey",
            "csv_delimiter": ",",
            "invalid_policy": "skip",
            "hnsw_m": 16,
            "hnsw_ef_construction": 100,
            "maintenance_work_mem_mb": 256,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    return factory


class FakeContext:
    """Small context manager used by database unit-test doubles."""

    def __init__(self, value):
        self.value = value

    def __enter__(self):
        return self.value

    def __exit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_context():
    return FakeContext

"""Small real-data tests for LibraryBuilder preparation without PostgreSQL."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ms2query_db.builder import LibraryBuilder
from ms2query_db.config import BuildConfig
from ms2query_db.embeddings import MS2DeepScoreEmbedder
from ms2query_db.io import Source, iter_source_batches

RESOURCES = Path(__file__).parent / "resources"
TEST_MGF = RESOURCES / "test_spectra.mgf"
TEST_COMPOUNDS = RESOURCES / "test_compounds.csv"
TEST_MODEL_PT = RESOURCES / "testmodel_v1.pt"


def _require(path: Path) -> Path:
    if not path.is_file():
        pytest.fail(f"Required test resource is missing: {path}")
    return path


def test_builder_prepares_real_reference_compounds_with_real_fingerprints():
    """Exercise CSV -> identity -> fingerprint -> projection using real chemistry code.

    ``LibraryBuilder._prepare()`` records row-local rejections immediately, but
    accepted rows receive their final ``RecordDecision`` only later, when
    ``Database.commit_batch()`` has resolved the database compound IDs.
    Therefore a preparation-stage test must not call ``ledger.counts()``.
    """
    pytest.importorskip("rdkit")
    pytest.importorskip("chemap")

    config = BuildConfig(
        schema="resource_builder_test",
        ionmode="both",
        classical_methods=(),
    )
    builder = LibraryBuilder(database=None, config=config)
    source = Source.from_path(_require(TEST_COMPOUNDS), "chemicals")
    records = [
        record
        for batch in iter_source_batches(
            source,
            100,
            smiles_column=config.smiles_column,
        )
        for record in batch
    ]

    ledger, prepared = builder._prepare(records, reference=False)

    # At this stage ``ledger`` contains only rows rejected during preparation;
    # accepted rows are represented by ``prepared`` and are finalized after DB
    # ID resolution in Database.commit_batch(). Together they must account for
    # every physical CSV record exactly once.
    assert len(records) == 100
    assert len(prepared) + len(ledger) == len(records)
    assert prepared
    assert all(row["spectrum"] is None for row in prepared)
    assert all(row["compound"]["identity_key"] for row in prepared)
    assert all(row["compound"]["fingerprint"] for row in prepared)
    assert all(
        len(row["compound"]["fingerprint_projection"])
        == config.fingerprints.projection_dim
        for row in prepared
    )


def test_builder_prepares_real_mgf_rows_with_real_ms2deepscore_embeddings():
    """Exercise MGF parsing, structure resolution, fingerprints, and model inference together."""
    pytest.importorskip("rdkit")
    pytest.importorskip("chemap")
    pytest.importorskip("ms2deepscore")

    config = BuildConfig(
        schema="resource_spectrum_builder_test",
        ionmode="both",
        classical_methods=(),
        batch_size=8,
    )
    embedder = MS2DeepScoreEmbedder.from_file(
        _require(TEST_MODEL_PT),
        allow_legacy=True,
        batch_size=8,
    )
    builder = LibraryBuilder(
        database=None,
        config=config,
        embedder=embedder,
    )
    source = Source.from_path(_require(TEST_MGF), "spectra")
    first_batch = next(iter_source_batches(source, config.batch_size))

    ledger, prepared = builder._prepare(first_batch, reference=True)

    # As above, successful rows are not entered into the ledger until the
    # database has assigned compound/spectrum IDs. No preparation-time
    # rejection is expected for this curated MGF fixture.
    assert len(ledger) == 0
    assert len(prepared) == len(first_batch)

    for item in prepared:
        spectrum = item["spectrum"]
        assert spectrum["ionmode"] in {"positive", "negative"}
        assert spectrum["n_peaks"] > 0
        assert spectrum["embedding"] is not None
        embedding = np.asarray(spectrum["embedding"])
        assert embedding.shape == (embedder.dimension,)
        assert np.isfinite(embedding).all()
        assert np.linalg.norm(embedding) == pytest.approx(1.0, rel=1e-5)

"""Optional real PostgreSQL/pgvector smoke test using repository test resources.

Set ``MS2QUERY_TEST_DSN`` to enable this module, for example to a disposable
PostgreSQL+pgvector service used by CI. The test creates and drops its own
random schema and never touches an existing MS2Query-DB library.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import numpy as np
import pytest
from sqlalchemy import text

from ms2query_db.builder import LibraryBuilder
from ms2query_db.config import BuildConfig
from ms2query_db.db.database import Database
from ms2query_db.embeddings import MS2DeepScoreEmbedder, validated_embeddings
from ms2query_db.errors import RejectedRecord
from ms2query_db.io import _mgf_records, load_mgf_batch
from ms2query_db.processing import strip_structure_annotations

RESOURCES = Path(__file__).parents[1] / "resources"
TEST_MGF = RESOURCES / "test_spectra.mgf"
TEST_COMPOUNDS = RESOURCES / "test_compounds.csv"
TEST_MODEL_PT = RESOURCES / "testmodel_v1.pt"


def _dsn() -> str:
    value = os.environ.get("MS2QUERY_TEST_DSN")
    if not value:
        pytest.skip("Set MS2QUERY_TEST_DSN to run PostgreSQL/pgvector integration tests")
    return value


def _resource(path: Path) -> Path:
    if not path.is_file():
        pytest.fail(f"Required test resource is missing: {path}")
    return path


def _load_collection():
    records = list(_mgf_records(_resource(TEST_MGF)))
    loaded = load_mgf_batch(records)
    assert loaded.rejected == {}
    assert loaded.collection is not None
    return loaded.collection


def _library_ionmode(collection) -> str:
    modes = {
        str(value).strip().lower()
        for value in collection.metadata["ionmode"]
    }
    assert modes <= {"positive", "negative"}
    assert modes
    return "both" if modes == {"positive", "negative"} else next(iter(modes))


def test_real_resource_library_build_and_embedding_search():
    """Build a tiny real library and verify mode-restricted pgvector search."""
    pytest.importorskip("psycopg")
    pytest.importorskip("ms2deepscore")
    pytest.importorskip("rdkit")
    pytest.importorskip("chemap")

    collection = _load_collection()
    schema = f"ms2q_test_{uuid.uuid4().hex[:12]}"
    config = BuildConfig(
        schema=schema,
        ionmode=_library_ionmode(collection),
        batch_size=16,
        classical_methods=(),
        hnsw_m=8,
        hnsw_ef_construction=32,
        maintenance_work_mem_mb=64,
    )
    embedder = MS2DeepScoreEmbedder.from_file(
        _resource(TEST_MODEL_PT),
        allow_legacy=True,
        batch_size=16,
    )
    database = Database(_dsn(), schema)

    try:
        builder = LibraryBuilder(
            database,
            config,
            embedder=embedder,
        )
        report = builder.build(
            _resource(TEST_MGF),
            chemical_csv=_resource(TEST_COMPOUNDS),
        )

        assert report.schema == schema
        assert report.counts["spectra"] == len(collection)
        assert report.counts["compounds"] >= 100
        assert report.counts["records_by_status"].get("rejected", 0) == 0

        stats = database.statistics()
        assert stats["spectra"] == len(collection)
        assert sum(stats["spectra_by_ionmode"].values()) == len(collection)

        # Use one real query embedding and require the ANN result to stay in the
        # same ion mode. Query annotations are stripped just like production
        # inference so structural labels cannot influence the model.
        query = strip_structure_annotations(collection[:1])
        values = validated_embeddings(embedder, query)
        assert len(values) == 1
        assert not isinstance(values[0], RejectedRecord)
        query_mode = str(query.metadata.iloc[0]["ionmode"]).strip().lower()

        hits = database.search_embeddings(
            [values[0]],
            top_n=3,
            ionmode=query_mode,
            ef_search=50,
        )[0]
        assert hits
        assert all(np.isfinite(hit["score"]) for hit in hits)

        candidates = database.get_candidates([hit["spectrum_id"] for hit in hits])
        assert {row["ionmode"] for row in candidates.values()} == {query_mode}
    finally:
        try:
            quoted = database.engine.dialect.identifier_preparer.quote_schema(schema)
            with database.engine.begin() as connection:
                connection.execute(text(f"DROP SCHEMA IF EXISTS {quoted} CASCADE"))
        finally:
            database.close()

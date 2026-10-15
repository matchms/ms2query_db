from __future__ import annotations

import zlib
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.sql import Select

import ms2query_db.db.database as database_module
from ms2query_db.db.database import Database
from ms2query_db.db.models import models_for
from ms2query_db.errors import CompatibilityError, ContractError
from ms2query_db.records import BatchLedger


def bare_database(*, ionmode="positive", fingerprint_size=16):
    database = object.__new__(Database)
    database.schema = "test_library"
    database.models = models_for(3, 4)
    database.config = SimpleNamespace(
        ionmode=ionmode,
        fingerprints=SimpleNamespace(size=fingerprint_size),
        maintenance_work_mem_mb=256,
        hnsw_m=16,
        hnsw_ef_construction=100,
    )
    return database


def test_database_requires_postgresql_dsn():
    with pytest.raises(ValueError, match="PostgreSQL"):
        Database("sqlite:///tmp.db", "test_library")


@pytest.mark.parametrize(
    ("library_mode", "requested", "expected"),
    [
        ("positive", None, "positive"),
        ("negative", None, "negative"),
        ("both", None, None),
        ("both", "positive", "positive"),
        ("both", "negative", "negative"),
    ],
)
def test_normalize_ionmode(library_mode, requested, expected):
    database = bare_database(ionmode=library_mode)
    assert database._normalize_ionmode(requested) == expected


def test_normalize_ionmode_rejects_mode_not_present_in_single_mode_library():
    database = bare_database(ionmode="positive")
    with pytest.raises(ValueError, match="not 'negative'"):
        database._normalize_ionmode("negative")


def test_embedding_index_modes_requires_both_modes_for_combined_library(monkeypatch):
    database = bare_database(ionmode="both")
    monkeypatch.setattr(
        database,
        "_spectra_by_ionmode",
        lambda: {"positive": 12},
    )

    with pytest.raises(CompatibilityError, match="missing"):
        database._embedding_index_modes()


def test_embedding_index_modes_returns_two_partial_index_modes(monkeypatch):
    database = bare_database(ionmode="both")
    monkeypatch.setattr(
        database,
        "_spectra_by_ionmode",
        lambda: {"positive": 12, "negative": 8},
    )

    assert database._embedding_index_modes() == ("positive", "negative")


def test_embedding_index_modes_rejects_unexpected_stored_mode(monkeypatch):
    database = bare_database(ionmode="both")
    monkeypatch.setattr(
        database,
        "_spectra_by_ionmode",
        lambda: {"positive": 1, "negative": 1, "both": 1},
    )

    with pytest.raises(ContractError, match="invalid ion modes"):
        database._embedding_index_modes()


def test_combined_library_requires_explicit_mode_for_approximate_embedding_search():
    database = bare_database(ionmode="both")

    with pytest.raises(NotImplementedError, match="cross-mode"):
        database.search_embeddings(
            [np.asarray([1.0, 0.0, 0.0])],
            5,
            ionmode=None,
            exact=False,
        )


class RecordingConnection:
    def __init__(self, *, select_rows=None, scalar_value=1):
        self.select_rows = select_rows or []
        self.scalar_value = scalar_value
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((statement, params))
        if isinstance(statement, Select):
            return list(self.select_rows)
        return []

    def scalar(self, statement, params=None):
        self.statements.append((statement, params))
        return self.scalar_value


class RecordingEngine:
    def __init__(self, connection):
        self.connection = connection
        self.dialect = postgresql.dialect()

    @contextmanager
    def begin(self):
        yield self.connection

    @contextmanager
    def connect(self):
        yield self.connection


def test_search_embeddings_applies_same_mode_sql_predicate():
    row = SimpleNamespace(id=17, distance=0.2)
    connection = RecordingConnection(select_rows=[row])
    database = bare_database(ionmode="both")
    database.engine = RecordingEngine(connection)

    hits = database.search_embeddings(
        [np.asarray([1.0, 0.0, 0.0])],
        5,
        ionmode="negative",
    )

    assert hits == [[{"spectrum_id": 17, "score": pytest.approx(0.8), "matches": None}]]
    selects = [stmt for stmt, _ in connection.statements if isinstance(stmt, Select)]
    assert len(selects) == 1
    sql = str(selects[0].compile(dialect=postgresql.dialect()))
    assert "reference_spectra.ionmode" in sql


def test_search_embeddings_validates_query_vector_before_sql_select():
    connection = RecordingConnection()
    database = bare_database(ionmode="positive")
    database.engine = RecordingEngine(connection)

    with pytest.raises(ValueError, match="finite"):
        database.search_embeddings(
            [np.asarray([1.0, np.nan, 0.0])],
            5,
        )


def test_finalize_builds_mode_specific_partial_hnsw_indexes(monkeypatch):
    connection = RecordingConnection(scalar_value=1)
    database = bare_database(ionmode="both")
    database.engine = RecordingEngine(connection)
    monkeypatch.setattr(
        database,
        "_embedding_index_modes",
        lambda: ("positive", "negative"),
    )
    states = []
    monkeypatch.setattr(database, "set_state", lambda state, error=None: states.append(state))

    database.finalize()

    sql = "\n".join(str(statement) for statement, _ in connection.statements)
    assert "spectrum_embedding_positive_hnsw" in sql
    assert "ionmode = 'positive' AND embedding IS NOT NULL" in sql
    assert "spectrum_embedding_negative_hnsw" in sql
    assert "ionmode = 'negative' AND embedding IS NOT NULL" in sql
    assert "chemical_projection_hnsw" in sql
    assert states == ["indexing", "ready"]


def test_commit_batch_resolves_compounds_spectra_and_finishes_one_transaction(
    monkeypatch,
):
    connection = object()
    database = bare_database()
    database.engine = RecordingEngine(connection)

    events = []

    class CompoundRepository:
        def __init__(self, conn, models):
            assert conn is connection

        def upsert_many(self, rows):
            events.append(("compounds", list(rows)))
            return {"A": 101, "B": 102}

    class SpectrumRepository:
        def __init__(self, conn, models):
            assert conn is connection

        def insert_many(self, rows):
            events.append(("spectra", list(rows)))
            return {5: 201, 6: 202}

    class ImportRepository:
        def __init__(self, conn, models):
            assert conn is connection

        def finish(self, source_id, ledger):
            events.append(("finish", source_id, ledger.finish()))

    monkeypatch.setattr(database_module, "ReferenceCompoundRepository", CompoundRepository)
    monkeypatch.setattr(database_module, "ReferenceSpectrumRepository", SpectrumRepository)
    monkeypatch.setattr(database_module, "ImportRepository", ImportRepository)

    ledger = BatchLedger(5, 2)
    prepared = [
        {
            "index": 5,
            "compound": {"identity_key": "A"},
            "spectrum": {"ionmode": "positive"},
            "detail": {"ionmode": "positive"},
        },
        {
            "index": 6,
            "compound": {"identity_key": "B"},
            "spectrum": {"ionmode": "positive"},
            "detail": {"ionmode": "positive"},
        },
    ]

    database.commit_batch(77, ledger, prepared, reference=True)

    decisions = ledger.finish()
    assert [(d.compound_id, d.spectrum_id) for d in decisions] == [
        (101, 201),
        (102, 202),
    ]
    assert events[-1][0] == "finish"


def test_decode_stored_fingerprint_retries_once_after_fresh_read(monkeypatch):
    database = bare_database(fingerprint_size=4)
    connection = RecordingConnection(scalar_value=b"fresh-payload")
    database.engine = RecordingEngine(connection)

    calls = []

    def fake_decode(payload, size):
        calls.append(payload)
        if len(calls) == 1:
            raise zlib.error("transient")
        assert size == 4
        return np.asarray([1.0, 2.0, 3.0, 4.0])

    monkeypatch.setattr(database_module, "decode_fingerprint", fake_decode)

    result = database._decode_stored_fingerprint(
        11,
        b"first-payload",
        spectrum_id=99,
    )

    np.testing.assert_array_equal(result, [1.0, 2.0, 3.0, 4.0])
    assert calls == [b"first-payload", b"fresh-payload"]


def test_decode_stored_fingerprint_raises_after_persistent_failure(monkeypatch):
    database = bare_database(fingerprint_size=4)
    connection = RecordingConnection(scalar_value=b"fresh-payload")
    database.engine = RecordingEngine(connection)

    monkeypatch.setattr(
        database_module,
        "decode_fingerprint",
        lambda payload, size: (_ for _ in ()).throw(zlib.error("bad")),
    )

    with pytest.raises(ContractError, match="could not be decoded"):
        database._decode_stored_fingerprint(11, b"first-payload")

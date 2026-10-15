"""Resource-backed tests for the public Python API boundary."""
from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace

import pytest

from ms2query_db.api import MS2QueryLibrary, _records_from_loaded_batch
from ms2query_db.config import BuildConfig, QueryConfig
from ms2query_db.io import discover_sources, iter_source_batches, load_mgf_batch
from ms2query_db.results import SpectrumQueryResult


RESOURCES = Path(__file__).parent / "resources"
TEST_SPECTRA = RESOURCES / "test_spectra.mgf"


def _require(path: Path) -> Path:
    assert path.is_file(), f"Missing Git-tracked test resource: {path}"
    return path


def test_records_from_loaded_batch_preserve_real_mgf_source_indices():
    """The API bridge must not renumber physical MGF records after parsing."""
    source = discover_sources(_require(TEST_SPECTRA))[0]
    raw_records = next(iter_source_batches(source, 4))
    loaded = load_mgf_batch(raw_records)

    records = _records_from_loaded_batch(loaded)

    assert [record.index for record in records] == [
        record.index for record in raw_records
    ]
    assert len(records) == len(raw_records)
    for record in records:
        if record.index in loaded.rejected:
            assert record.error is loaded.rejected[record.index]
            assert record.value is None
        else:
            assert record.error is None
            assert record.value is not None


class _RecordingEngine:
    def __init__(self):
        self.calls = []

    def process_batch(self, records, *, source_label="query"):
        records = list(records)
        self.calls.append((source_label, records))
        output = []
        for record in records:
            result = SpectrumQueryResult(
                f"{source_label}:{record.index}",
                record.index,
            )
            if record.error is not None:
                result.status = "rejected"
                result.reason_code = record.error.code
                result.reason = str(record.error)
            else:
                result.status = "no_hits"
            output.append(result)
        return output


def _recording_library() -> tuple[MS2QueryLibrary, _RecordingEngine]:
    """Construct only the API shell; MGF parsing remains fully real."""
    library = object.__new__(MS2QueryLibrary)
    library.config = BuildConfig(
        schema="api_resource_test",
        ionmode="both",
        classical_methods=(),
    )
    library.info = {
        "library_id": "resource-test-library",
        "descriptor": {"resource_test": True},
    }
    library.database = SimpleNamespace(schema="api_resource_test")
    engine = _RecordingEngine()

    def _engine(self, config, processor, reranker):
        return engine

    library._engine = MethodType(_engine, library)
    return library, engine


def test_iter_query_file_uses_real_mgf_parser_and_keeps_physical_query_ids():
    """File querying should preserve MGF record IDs across bounded batches."""
    library, engine = _recording_library()
    config = QueryConfig(
        search_methods=("entropy_hybrid",),
        batch_size=3,
        annotation_policy="ignore",
    )

    batches = list(
        library.iter_query_file(
            _require(TEST_SPECTRA),
            config=config,
        )
    )

    source = discover_sources(TEST_SPECTRA)[0]
    expected = [
        record.index
        for raw_batch in iter_source_batches(source, config.batch_size)
        for record in raw_batch
    ]
    observed = [
        result.query_index
        for batch in batches
        for result in batch.results
    ]

    assert observed == expected
    assert [call[0] for call in engine.calls] == [
        TEST_SPECTRA.name
    ] * len(engine.calls)
    assert all(
        batch.provenance["source"]["sha256"] == source.sha256
        for batch in batches
    )


def test_query_file_combines_resource_batches_without_changing_order():
    library, _engine = _recording_library()
    config = QueryConfig(
        search_methods=("entropy_hybrid",),
        batch_size=2,
        annotation_policy="ignore",
    )

    result = library.query_file(
        _require(TEST_SPECTRA),
        config=config,
    )

    assert [item.query_index for item in result.results] == sorted(
        item.query_index for item in result.results
    )
    assert result.provenance["source"]["name"] == TEST_SPECTRA.name
    assert result.provenance["query_config"] == config.to_dict()

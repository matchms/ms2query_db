"""Resource-backed smoke tests for real MGF/CSV parsing and matchms processing."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ms2query_db.config import ProcessingSpec
from ms2query_db.io import Source, _mgf_records, iter_source_batches, load_mgf_batch
from ms2query_db.processing import (
    make_processor,
    strip_structure_annotations,
    validate_collection,
)

RESOURCES = Path(__file__).parent / "resources"
TEST_MGF = RESOURCES / "test_spectra.mgf"
TEST_COMPOUNDS = RESOURCES / "test_compounds.csv"


def _resource(path: Path) -> Path:
    if not path.is_file():
        pytest.fail(f"Required test resource is missing: {path}")
    return path


def _load_test_spectra():
    records = list(_mgf_records(_resource(TEST_MGF)))
    loaded = load_mgf_batch(records)
    return records, loaded


def test_compound_resource_contains_100_real_rows_with_normalized_headers():
    """The small reference CSV should exercise the real CSV ingestion path."""
    source = Source.from_path(_resource(TEST_COMPOUNDS), "chemicals")
    batches = list(
        iter_source_batches(
            source,
            batch_size=17,
            smiles_column="SMILES",
        )
    )
    records = [record for batch in batches for record in batch]

    assert len(records) == 100
    assert [record.index for record in records] == list(range(100))
    assert all(record.error is None for record in records)

    expected_columns = {
        "inchikey",
        "smiles",
        "kingdom",
        "superclass",
        "class",
        "subclass",
    }
    assert set(records[0].value) == expected_columns
    assert all(record.value["smiles"].strip() for record in records)
    assert all(record.value["inchikey"].strip() for record in records)


def test_real_mgf_loads_to_collection_without_losing_physical_record_indices():
    """Real matchms parsing must preserve the source-record accounting contract."""
    records, loaded = _load_test_spectra()

    assert records
    assert loaded.rejected == {}
    assert loaded.collection is not None
    assert len(loaded.collection) == len(records)
    assert loaded.source_indices == tuple(record.index for record in records)

    metadata = loaded.collection.metadata
    assert "precursor_mz" in metadata.columns
    assert "ionmode" in metadata.columns

    modes = {str(value).strip().lower() for value in metadata["ionmode"]}
    assert modes
    assert modes <= {"positive", "negative"}


def test_real_collection_passes_standard_query_processing_and_validation():
    """Run the real matchms processor on the resource collection, not a test double."""
    _, loaded = _load_test_spectra()
    clean = strip_structure_annotations(loaded.collection)
    processor = make_processor(ProcessingSpec())

    processed = processor.process_collection(clean)

    assert processed is not None
    assert 0 < len(processed) <= len(clean)

    errors, precursor_mz, ionmodes = validate_collection(processed, "both")
    assert errors == [None] * len(processed)
    assert np.isfinite(precursor_mz).all()
    assert np.all(precursor_mz > 0)
    assert set(ionmodes) <= {"positive", "negative"}

    # Query processing normalizes intensities. At least one positive peak must
    # remain in every row after the standard pipeline.
    for row_index in range(len(processed)):
        mz, intensity = processed.fragments.get_row(row_index)
        assert len(mz) == len(intensity) >= ProcessingSpec().min_peaks
        assert np.isfinite(mz).all()
        assert np.isfinite(intensity).all()
        assert np.max(intensity) == pytest.approx(1.0)

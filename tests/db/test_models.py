from __future__ import annotations

import pytest
from sqlalchemy import CheckConstraint

from ms2query_db.db.models import models_for


def _check_constraints(table):
    return {
        constraint.name: str(constraint.sqltext)
        for constraint in table.constraints
        if isinstance(constraint, CheckConstraint)
    }


def test_models_for_is_cached_by_vector_dimensions():
    assert models_for(128, 64) is models_for(128, 64)
    assert models_for(128, 64) is not models_for(256, 64)


@pytest.mark.parametrize(
    ("embedding_dim", "projection_dim"),
    [(0, 64), (2001, 64), (128, 0), (128, 2001)],
)
def test_models_for_rejects_invalid_vector_dimensions(
    embedding_dim, projection_dim
):
    with pytest.raises(ValueError):
        models_for(embedding_dim, projection_dim)


def test_reference_spectrum_schema_enforces_real_ionmodes_and_payload_contracts():
    models = models_for(128, 64)
    table = models.ReferenceSpectrum.__table__
    constraints = _check_constraints(table)

    assert "ck_reference_spectrum_ionmode" in constraints
    assert "positive" in constraints["ck_reference_spectrum_ionmode"]
    assert "negative" in constraints["ck_reference_spectrum_ionmode"]
    assert "ck_reference_spectrum_mz_payload" in constraints
    assert "ck_reference_spectrum_intensity_payload" in constraints
    assert "ck_reference_spectrum_precursor" in constraints

    assert table.c.ionmode.type.length == 8
    assert table.c.ionmode.index is True
    assert table.c.compound_id.index is True


def test_library_and_import_tables_have_named_integrity_constraints():
    models = models_for(32, 16)

    library_constraints = _check_constraints(models.LibraryInfo.__table__)
    assert set(library_constraints) >= {
        "ck_library_info_singleton",
        "ck_library_info_state",
    }

    batch_constraints = _check_constraints(models.ImportBatch.__table__)
    assert set(batch_constraints) >= {
        "ck_import_batch_accounting",
        "ck_import_batch_nonnegative",
    }

    record_constraints = _check_constraints(models.ImportRecord.__table__)
    assert set(record_constraints) >= {
        "ck_import_record_status",
        "ck_import_record_decision",
    }

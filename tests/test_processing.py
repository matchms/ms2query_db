from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from ms2query_db.errors import ContractError, RejectedRecord
from ms2query_db.processing import (
    collection_from_records,
    process_query_collection,
    spectrum_from_record,
    strip_structure_annotations,
    validate_collection,
    validate_ionmode,
    validate_spectrum,
)


@pytest.mark.parametrize(
    ("value", "library_mode", "expected"),
    [
        ("positive", "positive", "positive"),
        ("negative", "negative", "negative"),
        ("positive", "both", "positive"),
        ("negative", "both", "negative"),
        (" POSITIVE ", "both", "positive"),
    ],
)
def test_validate_ionmode_accepts_library_compatible_modes(
    value, library_mode, expected
):
    assert validate_ionmode(value, library_mode) == expected


@pytest.mark.parametrize("value", [None, "", "both", "unknown"])
def test_validate_ionmode_rejects_invalid_spectrum_modes(value):
    with pytest.raises(RejectedRecord) as exc:
        validate_ionmode(value, "both")
    assert exc.value.code == "invalid_ionmode"


def test_validate_ionmode_rejects_single_mode_mismatch():
    with pytest.raises(RejectedRecord) as exc:
        validate_ionmode("negative", "positive")
    assert exc.value.code == "ionmode_mismatch"


def test_strip_structure_annotations_collection_is_non_mutating(make_spectrum):
    matchms = pytest.importorskip("matchms")
    collection = matchms.SpectraCollection(
        [
            make_spectrum(
                smiles="CCO",
                inchikey="LFQSCWFLJHTTHZ-UHFFFAOYSA-N",
                sample="keep-me",
            ),
            make_spectrum(
                smiles="CCN",
                superclass="Organic compounds",
                sample="also-keep",
            ),
        ],
        mz_precision=1e-6,
    )

    stripped = strip_structure_annotations(collection)

    assert stripped is not collection
    assert "smiles" in collection.metadata.columns
    assert "smiles" not in stripped.metadata.columns
    assert "inchikey" not in stripped.metadata.columns
    assert "superclass" not in stripped.metadata.columns
    assert stripped.metadata["sample"].tolist() == ["keep-me", "also-keep"]
    np.testing.assert_allclose(
        stripped.fragments.get_row(0)[0],
        collection.fragments.get_row(0)[0],
    )


def test_strip_structure_annotations_spectrum_is_non_mutating(make_spectrum):
    spectrum = make_spectrum(smiles="CCO", sample="x")

    stripped = strip_structure_annotations(spectrum)

    assert stripped is not spectrum
    assert spectrum.get("smiles") == "CCO"
    assert stripped.get("smiles") is None
    assert stripped.get("sample") == "x"
    np.testing.assert_allclose(stripped.peaks.mz, spectrum.peaks.mz)


def test_validate_spectrum_returns_float64_and_actual_mode(make_spectrum):
    spectrum = make_spectrum(ionmode="negative", precursor_mz="321.5")

    mz, intensity, precursor = validate_spectrum(spectrum, "both")

    assert mz.dtype == np.float64
    assert intensity.dtype == np.float64
    assert precursor == pytest.approx(321.5)


def test_validate_collection_rejects_rows_independently(make_spectrum):
    matchms = pytest.importorskip("matchms")
    collection = matchms.SpectraCollection(
        [
            make_spectrum(precursor_mz=100.0, ionmode="positive"),
            make_spectrum(precursor_mz="not-a-number", ionmode="positive"),
            make_spectrum(precursor_mz=300.0, ionmode="negative"),
        ],
        mz_precision=1e-6,
    )

    errors, precursors, modes = validate_collection(collection, "positive")

    assert errors[0] is None
    assert errors[1].code == "invalid_precursor"
    assert errors[2].code == "ionmode_mismatch"
    assert precursors[0] == pytest.approx(100.0)
    assert np.isnan(precursors[1])
    assert np.isnan(precursors[2])
    assert modes == ("positive", "positive", "negative")


def test_process_query_collection_preserves_input_alignment_when_filter_drops_rows(
    make_spectrum,
):
    class DropProcessor:
        def process_collection(self, collection):
            return collection.filter(collection.metadata["keep"].to_numpy(dtype=bool))

    spectra = [
        make_spectrum(smiles="CCO", keep=True, precursor_mz=101.0),
        make_spectrum(smiles="CCN", keep=False, precursor_mz=202.0),
        make_spectrum(smiles="CCC", keep=True, precursor_mz=303.0),
    ]

    result = process_query_collection(
        spectra,
        SimpleNamespace(),
        library_ionmode="positive",
        mz_precision=1e-6,
        processor=DropProcessor(),
    )

    assert result.input_rows == (0, 2)
    np.testing.assert_allclose(result.precursor_mz, [101.0, 303.0])
    assert result.ionmodes == ("positive", "positive")
    assert result.rejected[1].code == "filtered_spectrum"
    assert "smiles" not in result.spectra.metadata.columns
    assert "ms2query_input_row" not in result.spectra.metadata.columns


def test_process_query_collection_bisects_processor_errors_to_one_row(make_spectrum):
    class ExplodingProcessor:
        def process_collection(self, collection):
            if collection.metadata["explode"].fillna(False).any():
                raise ValueError("synthetic bad row")
            return collection

    spectra = [
        make_spectrum(explode=False, precursor_mz=101.0),
        make_spectrum(explode=True, precursor_mz=202.0),
        make_spectrum(explode=False, precursor_mz=303.0),
    ]

    result = process_query_collection(
        spectra,
        SimpleNamespace(),
        library_ionmode="positive",
        mz_precision=1e-6,
        processor=ExplodingProcessor(),
    )

    assert result.input_rows == (0, 2)
    assert result.rejected[1].code == "query_processing"
    assert "synthetic bad row" in str(result.rejected[1])
    assert len(result.spectra) == 2


def test_process_query_collection_empty_input_is_well_defined():
    result = process_query_collection(
        [],
        SimpleNamespace(),
        library_ionmode="both",
        mz_precision=1e-6,
        processor=SimpleNamespace(process_collection=lambda value: value),
    )

    assert result.spectra is None
    assert result.input_rows == ()
    assert result.ionmodes == ()
    assert result.rejected == {}
    assert result.precursor_mz.shape == (0,)


def _stored_row(*, n_peaks=3, ionmode="positive"):
    mz = np.asarray([50.0, 100.0, 150.0], dtype="<f8")
    intensity = np.asarray([1.0, 0.5, 0.25], dtype="<f8")
    return {
        "n_peaks": n_peaks,
        "mz_values": mz.tobytes(),
        "intensity_values": intensity.tobytes(),
        "precursor_mz": 200.0,
        "ionmode": ionmode,
        "other_metadata": {"sample": "reference"},
    }


def test_spectrum_from_record_reconstructs_canonical_payload():
    spectrum = spectrum_from_record(_stored_row())

    np.testing.assert_allclose(spectrum.peaks.mz, [50.0, 100.0, 150.0])
    np.testing.assert_allclose(spectrum.peaks.intensities, [1.0, 0.5, 0.25])
    assert spectrum.get("precursor_mz") == pytest.approx(200.0)
    assert spectrum.get("ionmode") == "positive"
    assert spectrum.get("sample") == "reference"


def test_spectrum_from_record_rejects_payload_length_mismatch():
    with pytest.raises(ContractError, match="payload length"):
        spectrum_from_record(_stored_row(n_peaks=4))


def test_collection_from_records_handles_empty_and_multiple_rows():
    assert collection_from_records([], mz_precision=1e-6) is None

    collection = collection_from_records(
        [_stored_row(), _stored_row(ionmode="negative")],
        mz_precision=1e-6,
    )
    assert len(collection) == 2
    assert collection.metadata["ionmode"].tolist() == ["positive", "negative"]

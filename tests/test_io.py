from __future__ import annotations

from pathlib import Path

import pytest

from ms2query_db.errors import ContractError, RejectedRecord
from ms2query_db.io import (
    LoadedSpectrumBatch,
    Source,
    _csv_records,
    _mgf_records,
    discover_sources,
    iter_source_batches,
    load_mgf_batch,
)
from ms2query_db.records import InputRecord


def mgf_record(title: str, *, body: str = "50 1\n") -> str:
    return (
        "BEGIN IONS\n"
        f"TITLE={title}\n"
        "PEPMASS=100\n"
        f"{body}"
        "END IONS\n"
    )


class SizedCollection:
    """Minimal collection double for loader-alignment tests."""

    def __init__(self, size):
        self.size = size

    def __len__(self):
        return self.size


def test_source_from_path_and_descriptor(tmp_path):
    path = tmp_path / "library.mgf"
    path.write_text(mgf_record("a"), encoding="utf-8")

    source = Source.from_path(path, "spectra")

    assert source.path == path.resolve()
    assert source.kind == "spectra"
    assert len(source.sha256) == 64
    assert source.descriptor() == {
        "name": "library.mgf",
        "sha256": source.sha256,
        "kind": "spectra",
        "bytes": path.stat().st_size,
    }


@pytest.mark.parametrize("kind", ["other", "", None])
def test_source_rejects_unknown_kind(tmp_path, kind):
    path = tmp_path / "x.mgf"
    path.write_text(mgf_record("x"), encoding="utf-8")

    with pytest.raises(ValueError, match="source kind"):
        Source(path, "0" * 64, kind)


def test_discover_sources_is_sorted_and_content_deduplicated(tmp_path):
    (tmp_path / "b.mgf").write_text(mgf_record("b"), encoding="utf-8")
    (tmp_path / "a.mgf").write_text(mgf_record("a"), encoding="utf-8")
    (tmp_path / "c.mgf").write_text(mgf_record("a"), encoding="utf-8")

    sources = discover_sources(tmp_path)

    # a.mgf and c.mgf are byte-identical, so only the first sorted path remains.
    assert [source.path.name for source in sources] == ["a.mgf", "b.mgf"]


def test_discover_sources_respects_recursive_flag(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "x.mgf").write_text(mgf_record("x"), encoding="utf-8")

    with pytest.raises(ValueError, match="No files"):
        discover_sources(tmp_path, recursive=False)

    sources = discover_sources(tmp_path, recursive=True)
    assert [source.path.name for source in sources] == ["x.mgf"]


def test_discover_sources_rejects_non_mgf_spectrum_file(tmp_path):
    path = tmp_path / "library.mzML"
    path.write_text("not an mgf", encoding="utf-8")

    with pytest.raises(ValueError, match="MGF"):
        discover_sources(path, kind="spectra")


def test_mgf_records_preserve_preamble_and_physical_indices(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text(
        "# global comment\n"
        + mgf_record("first")
        + "; between-record comment\n"
        + mgf_record("second"),
        encoding="utf-8",
    )

    records = list(_mgf_records(path))

    assert [record.index for record in records] == [0, 1]
    assert all(record.error is None for record in records)
    assert records[0].value.startswith("# global comment\nBEGIN IONS")
    assert "TITLE=first" in records[0].value
    assert "TITLE=second" in records[1].value


def test_mgf_records_nested_begin_is_row_local_and_alignment_continues(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text(
        "BEGIN IONS\nTITLE=broken\n"
        "BEGIN IONS\nTITLE=recovered\n50 1\nEND IONS\n",
        encoding="utf-8",
    )

    records = list(_mgf_records(path))

    assert [record.index for record in records] == [0, 1]
    assert records[0].error.code == "malformed_mgf"
    assert records[1].error is None
    assert "TITLE=recovered" in records[1].value


def test_mgf_records_missing_final_end_is_row_local_rejection(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text("BEGIN IONS\nTITLE=x\n50 1\n", encoding="utf-8")

    records = list(_mgf_records(path))

    assert len(records) == 1
    assert records[0].index == 0
    assert records[0].error.code == "malformed_mgf"
    assert "Missing final END IONS" in str(records[0].error)


def test_mgf_records_unmatched_end_is_fatal_file_error(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text("END IONS\n", encoding="utf-8")

    with pytest.raises(ValueError, match="END IONS without BEGIN IONS"):
        list(_mgf_records(path))


def test_mgf_records_unexpected_data_after_record_is_fatal(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text(mgf_record("x") + "garbage\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected data outside"):
        list(_mgf_records(path))


def test_csv_records_normalize_headers_and_reject_extra_fields(tmp_path):
    path = tmp_path / "compounds.csv"
    path.write_text(
        " SMILES ,Name\n"
        "CCO,Ethanol\n"
        "CCN,Ethylamine,unexpected\n",
        encoding="utf-8",
    )

    records = list(_csv_records(path, ",", "smiles"))

    assert records[0].value == {"smiles": "CCO", "name": "Ethanol"}
    assert records[1].error.code == "malformed_csv"


def test_csv_records_require_case_insensitively_unique_headers(tmp_path):
    path = tmp_path / "compounds.csv"
    path.write_text("SMILES, smiles\nCCO,CCO\n", encoding="utf-8")

    with pytest.raises(ValueError, match="unique column names"):
        list(_csv_records(path, ",", "smiles"))


def test_csv_records_require_configured_smiles_column(tmp_path):
    path = tmp_path / "compounds.csv"
    path.write_text("inchi,name\nInChI=1S/x,x\n", encoding="utf-8")

    with pytest.raises(ValueError, match="lacks configured SMILES"):
        list(_csv_records(path, ",", "smiles"))


def test_iter_source_batches_preserves_consecutive_record_groups(tmp_path):
    path = tmp_path / "x.mgf"
    path.write_text(
        mgf_record("0") + mgf_record("1") + mgf_record("2"),
        encoding="utf-8",
    )
    source = Source.from_path(path, "spectra")

    batches = list(iter_source_batches(source, 2))

    assert [[record.index for record in batch] for batch in batches] == [[0, 1], [2]]


@pytest.mark.parametrize("batch_size", [0, -1, True, 1.5])
def test_iter_source_batches_requires_positive_integer_batch_size(tmp_path, batch_size):
    path = tmp_path / "x.mgf"
    path.write_text(mgf_record("x"), encoding="utf-8")
    source = Source.from_path(path, "spectra")

    with pytest.raises(ValueError, match="positive integer"):
        list(iter_source_batches(source, batch_size))


def test_loaded_spectrum_batch_enforces_alignment_contract():
    with pytest.raises(ContractError, match="duplicate source indices"):
        LoadedSpectrumBatch(SizedCollection(2), (1, 1), {})

    with pytest.raises(ContractError, match="both parsed and rejected"):
        LoadedSpectrumBatch(
            SizedCollection(1),
            (1,),
            {1: RejectedRecord("x", "bad")},
        )

    with pytest.raises(ContractError, match="row count differs"):
        LoadedSpectrumBatch(SizedCollection(1), (1, 2), {})


def test_load_mgf_batch_empty_input_returns_empty_result():
    result = load_mgf_batch([], loader=lambda *args, **kwargs: None)

    assert result.collection is None
    assert result.source_indices == ()
    assert result.rejected == {}


def test_load_mgf_batch_uses_one_parser_call_for_valid_batch():
    records = [
        InputRecord(10, mgf_record("a")),
        InputRecord(11, mgf_record("b")),
        InputRecord(12, mgf_record("c")),
    ]
    calls = []

    def loader(path, **kwargs):
        text = Path(path).read_text(encoding="utf-8")
        calls.append((text, kwargs))
        return SizedCollection(text.count("BEGIN IONS"))

    result = load_mgf_batch(records, mz_precision=1e-5, loader=loader)

    assert len(calls) == 1
    assert calls[0][1] == {
        "metadata_harmonization": True,
        "ftype": "mgf",
        "mz_precision": 1e-5,
    }
    assert result.source_indices == (10, 11, 12)
    assert result.rejected == {}
    assert len(result.collection) == 3


def test_load_mgf_batch_does_not_reparse_pre_rejected_rows():
    records = [
        InputRecord(0, mgf_record("good")),
        InputRecord(1, error=RejectedRecord("malformed_mgf", "bad boundary")),
    ]
    seen = []

    def loader(path, **kwargs):
        text = Path(path).read_text(encoding="utf-8")
        seen.append(text)
        return SizedCollection(1)

    result = load_mgf_batch(records, loader=loader)

    assert len(seen) == 1
    assert "good" in seen[0]
    assert "bad boundary" not in seen[0]
    assert result.source_indices == (0,)
    assert result.rejected[1].code == "malformed_mgf"


def test_load_mgf_batch_bisects_parser_failure_and_preserves_source_order(
    make_spectrum,
):
    matchms = pytest.importorskip("matchms")
    records = [
        InputRecord(20, mgf_record("a")),
        InputRecord(21, mgf_record("BAD")),
        InputRecord(22, mgf_record("c")),
        InputRecord(23, mgf_record("d")),
    ]
    calls = []

    def loader(path, **kwargs):
        text = Path(path).read_text(encoding="utf-8")
        calls.append(text)
        if "TITLE=BAD" in text:
            raise ValueError("synthetic parser failure")
        titles = [
            line.split("=", 1)[1]
            for line in text.splitlines()
            if line.startswith("TITLE=")
        ]
        spectra = [
            make_spectrum(title=title, precursor_mz=100 + index)
            for index, title in enumerate(titles)
        ]
        return matchms.SpectraCollection(spectra, mz_precision=1e-6)

    result = load_mgf_batch(records, loader=loader)

    assert len(calls) > 1
    assert result.source_indices == (20, 22, 23)
    assert result.rejected[21].code == "spectrum_parse"
    assert len(result.collection) == 3
    assert result.collection.metadata["title"].tolist() == ["a", "c", "d"]


def test_load_mgf_batch_row_count_mismatch_is_isolated_to_single_records():
    records = [
        InputRecord(0, mgf_record("a")),
        InputRecord(1, mgf_record("b")),
    ]

    def loader(path, **kwargs):
        return SizedCollection(0)

    result = load_mgf_batch(records, loader=loader)

    assert result.collection is None
    assert result.source_indices == ()
    assert set(result.rejected) == {0, 1}
    assert all(error.code == "spectrum_parse" for error in result.rejected.values())


def test_load_mgf_batch_rejects_duplicate_input_indices():
    records = [
        InputRecord(7, mgf_record("a")),
        InputRecord(7, mgf_record("b")),
    ]

    with pytest.raises(ContractError, match="duplicate source indices"):
        load_mgf_batch(records, loader=lambda *args, **kwargs: SizedCollection(2))

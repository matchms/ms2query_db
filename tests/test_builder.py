from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

import ms2query_db.builder as builder_module
from ms2query_db.builder import LibraryBuilder, check_metadata, compound_metadata
from ms2query_db.errors import ContractError, RejectedRecord
from ms2query_db.io import LoadedSpectrumBatch
from ms2query_db.records import BatchLedger, InputRecord


@dataclass(frozen=True)
class DummyIdentity:
    identity_key: str
    smiles: str
    inchikey14: str = "ABCDEFGHIJKLMN"
    inchikey: str = "ABCDEFGHIJKLMN-ABCDEFGHIJ-A"
    inchi: str = "InChI=1S/test"
    formula: str = "C2H6O"
    exact_mass: float = 46.0419

    def to_dict(self):
        return {
            "identity_key": self.identity_key,
            "inchikey14": self.inchikey14,
            "inchikey": self.inchikey,
            "smiles": self.smiles,
            "inchi": self.inchi,
            "formula": self.formula,
            "exact_mass": self.exact_mass,
        }


def bare_builder(config, *, fingerprinter=None, embedder=None, spectrum_loader=None):
    """Construct LibraryBuilder without creating a real projector."""
    builder = object.__new__(LibraryBuilder)
    builder.database = None
    builder.config = config
    builder.embedder = embedder
    builder.fingerprinter = fingerprinter
    builder.spectrum_loader = spectrum_loader or (lambda records, mz_precision: records)
    builder.spectrum_stripper = lambda collection: collection
    builder.projector = None
    builder.resumed_batches = 0
    return builder


def test_check_metadata_rejects_nested_nul_text():
    with pytest.raises(RejectedRecord) as exc:
        check_metadata({"safe": ["ok", {"bad": "hello\x00world"}]})
    assert exc.value.code == "invalid_metadata"


def test_compound_metadata_splits_typed_and_extra_fields(caplog):
    row = {
        "SMILES": "CCO",
        "Class": "Alcohols",
        "Superclass": "Organic compounds",
        "visual_x": "1.25",
        "visual_y": "broken",
        "pubchem_cid": "702",
        "instrument": "Orbitrap",
    }

    result = compound_metadata(row)

    assert result["compound_class"] == "Alcohols"
    assert result["superclass"] == "Organic compounds"
    assert result["visual_x"] == pytest.approx(1.25)
    assert result["visual_y"] is None
    assert result["pubchem_cid"] == 702
    assert result["extra_metadata"]["instrument"] == "Orbitrap"
    assert result["extra_metadata"]["visual_y"] == "broken"
    assert "smiles" not in result["extra_metadata"]
    assert "Ignoring invalid optional visual_y" in caplog.text


def test_reject_raise_policy_preserves_rejection_code(build_config_factory):
    builder = bare_builder(
        build_config_factory(invalid_policy="raise"),
        fingerprinter=SimpleNamespace(),
    )
    ledger = BatchLedger(7, 1)

    with pytest.raises(RejectedRecord) as exc:
        builder._reject(ledger, 7, RejectedRecord("bad_row", "broken"))

    assert exc.value.code == "bad_row"
    assert "Source record 7" in str(exc.value)


def test_require_contiguous_batch_rejects_gaps(build_config_factory):
    builder = bare_builder(build_config_factory(), fingerprinter=SimpleNamespace())
    records = [InputRecord(10, {}), InputRecord(12, {})]
    ledger = BatchLedger(10, 2)

    with pytest.raises(ContractError, match="contiguous"):
        builder._require_contiguous_batch(records, ledger)


def test_prepare_reference_records_stores_actual_modes_for_combined_library(
    monkeypatch, make_spectrum, build_config_factory
):
    config = build_config_factory(ionmode="both")
    builder = bare_builder(config, fingerprinter=SimpleNamespace())

    monkeypatch.setattr(
        builder_module,
        "identify_compound",
        lambda smiles, inchi, inchikey: DummyIdentity(
            identity_key=smiles,
            smiles=smiles,
        ),
    )

    spectra = [
        make_spectrum(smiles="CCO", ionmode="positive"),
        make_spectrum(smiles="CCN", ionmode="negative"),
    ]
    matchms = pytest.importorskip("matchms")
    collection = matchms.SpectraCollection(spectra, mz_precision=1e-6)
    builder.spectrum_loader = lambda records, mz_precision: LoadedSpectrumBatch(
        collection=collection,
        source_indices=tuple(record.index for record in records),
        rejected={},
    )
    records = [InputRecord(0, "raw-0"), InputRecord(1, "raw-1")]
    ledger = BatchLedger(0, 2)

    pending, parsed_collection = builder._prepare_reference_records(records, ledger)

    assert len(parsed_collection) == 2
    assert [row["spectrum"]["ionmode"] for row in pending] == [
        "positive",
        "negative",
    ]
    assert [row["detail"]["ionmode"] for row in pending] == [
        "positive",
        "negative",
    ]
    assert all(row["spectrum"]["ionmode"] != "both" for row in pending)


def test_prepare_reference_records_rejects_wrong_mode_row_only(
    monkeypatch, make_spectrum, build_config_factory
):
    config = build_config_factory(ionmode="positive")
    builder = bare_builder(config, fingerprinter=SimpleNamespace())
    monkeypatch.setattr(
        builder_module,
        "identify_compound",
        lambda smiles, inchi, inchikey: DummyIdentity(smiles, smiles),
    )

    spectra = [
        make_spectrum(smiles="CCO", ionmode="positive"),
        make_spectrum(smiles="CCN", ionmode="negative"),
    ]
    matchms = pytest.importorskip("matchms")
    collection = matchms.SpectraCollection(spectra, mz_precision=1e-6)
    builder.spectrum_loader = lambda records, mz_precision: LoadedSpectrumBatch(
        collection=collection,
        source_indices=tuple(record.index for record in records),
        rejected={},
    )
    records = [InputRecord(0, "raw-0"), InputRecord(1, "raw-1")]
    ledger = BatchLedger(0, 2)

    pending, _ = builder._prepare_reference_records(records, ledger)

    assert [row["index"] for row in pending] == [0]
    assert ledger._decisions[1].reason_code == "ionmode_mismatch"


def test_attach_fingerprints_computes_duplicate_structure_once(build_config_factory):
    class Fingerprinter:
        def __init__(self):
            self.calls = []

        def compute_many(self, smiles):
            self.calls.append(list(smiles))
            return [
                np.asarray([1.0, 2.0]),
                np.asarray([3.0, 4.0]),
            ]

    fingerprinter = Fingerprinter()
    builder = bare_builder(build_config_factory(), fingerprinter=fingerprinter)
    a = DummyIdentity("A", "CCO")
    b = DummyIdentity("B", "CCN")
    pending = [
        {"index": 0, "identity": a},
        {"index": 1, "identity": a},
        {"index": 2, "identity": b},
    ]
    ledger = BatchLedger(0, 3)

    accepted = builder._attach_fingerprints(pending, ledger)

    assert fingerprinter.calls == [["CCO", "CCN"]]
    assert len(accepted) == 3
    np.testing.assert_array_equal(accepted[0]["fingerprint"], [1.0, 2.0])
    np.testing.assert_array_equal(accepted[1]["fingerprint"], [1.0, 2.0])
    np.testing.assert_array_equal(accepted[2]["fingerprint"], [3.0, 4.0])


def test_attach_fingerprints_rejects_all_rows_of_failed_structure(
    build_config_factory,
):
    class Fingerprinter:
        def compute_many(self, smiles):
            return [RejectedRecord("empty_fingerprint", "no fingerprint")]

    builder = bare_builder(
        build_config_factory(),
        fingerprinter=Fingerprinter(),
    )
    identity = DummyIdentity("A", "CCO")
    pending = [
        {"index": 0, "identity": identity},
        {"index": 1, "identity": identity},
    ]
    ledger = BatchLedger(0, 2)

    assert builder._attach_fingerprints(pending, ledger) == []
    assert ledger._decisions[0].reason_code == "empty_fingerprint"
    assert ledger._decisions[1].reason_code == "empty_fingerprint"


def test_attach_embeddings_is_collection_aligned_and_row_local(
    monkeypatch, make_spectrum, build_config_factory
):
    matchms = pytest.importorskip("matchms")
    collection = matchms.SpectraCollection(
        [
            make_spectrum(precursor_mz=100.0),
            make_spectrum(precursor_mz=200.0),
            make_spectrum(precursor_mz=300.0),
        ],
        mz_precision=1e-6,
    )

    builder = bare_builder(
        build_config_factory(),
        fingerprinter=SimpleNamespace(),
        embedder=SimpleNamespace(),
    )
    accepted = [
        {"index": 0, "collection_row": 0, "spectrum": {"embedding": None}},
        {"index": 2, "collection_row": 2, "spectrum": {"embedding": None}},
    ]
    ledger = BatchLedger(0, 3)

    seen = {}

    def fake_embeddings(embedder, subset):
        seen["precursors"] = subset.metadata["precursor_mz"].tolist()
        return [
            np.asarray([1.0, 0.0]),
            RejectedRecord("invalid_embedding", "bad vector"),
        ]

    monkeypatch.setattr(builder_module, "validated_embeddings", fake_embeddings)

    valid = builder._attach_embeddings(accepted, collection, ledger)

    assert seen["precursors"] == [100.0, 300.0]
    assert len(valid) == 1
    np.testing.assert_array_equal(valid[0]["spectrum"]["embedding"], [1.0, 0.0])
    assert ledger._decisions[2].reason_code == "invalid_embedding"


def test_materialize_compound_rows_preserves_first_metadata_row_and_projection(
    monkeypatch, build_config_factory
):
    class Projector:
        def transform(self, values):
            assert values.shape == (2, 2)
            return np.asarray([[0.1, 0.2], [0.3, 0.4]])

    builder = bare_builder(build_config_factory(), fingerprinter=SimpleNamespace())
    builder.projector = Projector()
    monkeypatch.setattr(builder_module, "encode_fingerprint", lambda fp: tuple(fp))

    a = DummyIdentity("A", "CCO")
    b = DummyIdentity("B", "CCN")
    accepted = [
        {"identity": a, "fingerprint": np.asarray([1.0, 2.0])},
        {"identity": a, "fingerprint": np.asarray([9.0, 9.0])},
        {"identity": b, "fingerprint": np.asarray([3.0, 4.0])},
    ]

    rows = builder._materialize_compound_rows(accepted)

    assert set(rows) == {"A", "B"}
    assert rows["A"]["fingerprint"] == (1.0, 2.0)
    np.testing.assert_allclose(rows["A"]["fingerprint_projection"], [0.1, 0.2])

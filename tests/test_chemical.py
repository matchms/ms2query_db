from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ms2query_db.search.chemical as chemical_module
from ms2query_db.errors import ContractError
from ms2query_db.search.chemical import ChemicalSearch


def make_search(size=2):
    database = SimpleNamespace(
        config=SimpleNamespace(
            fingerprints=SimpleNamespace(size=size)
        )
    )
    projector = SimpleNamespace(transform=lambda x: x)
    return ChemicalSearch(database, projector)


def candidate_rows():
    return [
        {"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.8},
        {"spectrum_id": 11, "compound_id": 1, "rf_quality": 0.6},
        {"spectrum_id": 20, "compound_id": 2, "rf_quality": 0.6},
        {"spectrum_id": 30, "compound_id": 3, "rf_quality": 0.4},
    ]


def stored_records():
    return {
        10: {
            "fingerprint": np.asarray([1.0, 0.0]),
            "compound_id": 1,
            "identity_key": "A",
        },
        11: {
            "fingerprint": np.asarray([1.0, 0.0]),
            "compound_id": 1,
            "identity_key": "A",
        },
        20: {
            "fingerprint": np.asarray([0.0, 1.0]),
            "compound_id": 2,
            "identity_key": "B",
        },
        30: {
            "fingerprint": np.asarray([1.0, 1.0]),
            "compound_id": 3,
            "identity_key": "C",
        },
    }


def test_compound_entries_use_max_replicate_support_and_power_two_weights():
    search = make_search()

    weighted = search._compound_entries(
        candidate_rows(),
        stored_records(),
        weight_source="rf_quality",
        selection="top_n",
        top_n=2,
        score_delta=0.2,
        weight_power=2.0,
    )

    entries = [entry for entry, _weight in weighted]
    weights = np.asarray([weight for _entry, weight in weighted])

    assert [entry.compound_id for entry in entries] == [1, 2]
    assert entries[0].spectrum_id == 10  # 0.8 wins over replicate score 0.6

    expected = np.asarray([0.8**2, 0.6**2], dtype=float)
    expected /= expected.sum()
    np.testing.assert_allclose(weights, expected)


def test_compound_entries_score_delta_selects_only_near_best_compounds():
    search = make_search()
    rows = [
        {"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.90},
        {"spectrum_id": 20, "compound_id": 2, "rf_quality": 0.80},
        {"spectrum_id": 30, "compound_id": 3, "rf_quality": 0.70},
    ]

    weighted = search._compound_entries(
        rows,
        stored_records(),
        weight_source="rf_quality",
        selection="score_delta",
        top_n=20,
        score_delta=0.15,
        weight_power=2.0,
    )

    assert [entry.compound_id for entry, _weight in weighted] == [1, 2]
    assert sum(weight for _entry, weight in weighted) == pytest.approx(1.0)


def test_compound_entries_can_use_spectral_score_as_weight_source():
    search = make_search()
    rows = [
        {
            "spectrum_id": 10,
            "compound_id": 1,
            "score_entropy_hybrid": 0.7,
        },
        {
            "spectrum_id": 20,
            "compound_id": 2,
            "score_entropy_hybrid": 0.9,
        },
    ]

    weighted = search._compound_entries(
        rows,
        stored_records(),
        weight_source="entropy_hybrid",
        selection="top_n",
        top_n=2,
        score_delta=0.2,
        weight_power=1.0,
    )

    assert [entry.compound_id for entry, _weight in weighted] == [2, 1]


def test_compound_entries_skip_nonpositive_or_nonfinite_support():
    search = make_search()
    rows = [
        {"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.0},
        {"spectrum_id": 20, "compound_id": 2, "rf_quality": -1.0},
        {"spectrum_id": 30, "compound_id": 3, "rf_quality": np.nan},
    ]

    assert search._compound_entries(
        rows,
        stored_records(),
        weight_source="rf_quality",
        selection="top_n",
        top_n=20,
        score_delta=0.2,
        weight_power=2.0,
    ) == []


def test_compound_entries_reject_candidate_without_stored_record():
    search = make_search()

    with pytest.raises(ContractError, match="no stored record"):
        search._compound_entries(
            [{"spectrum_id": 999, "compound_id": 1, "rf_quality": 0.8}],
            stored_records(),
            weight_source="rf_quality",
            selection="top_n",
            top_n=20,
            score_delta=0.2,
            weight_power=2.0,
        )


def test_weighted_medoid_selects_real_candidate_with_highest_centrality(monkeypatch):
    search = make_search()
    rows = [
        {"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.70},
        {"spectrum_id": 20, "compound_id": 2, "rf_quality": 0.69},
        {"spectrum_id": 30, "compound_id": 3, "rf_quality": 0.68},
    ]
    monkeypatch.setattr(
        chemical_module,
        "pairwise_similarity",
        lambda fingerprints, metric: np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ]
        ),
    )

    result = search.aggregate_candidates(
        rows,
        stored_records(),
        method="medoid",
        weight_source="rf_quality",
        weight_power=2.0,
        selection="top_n",
        top_n=3,
        score_delta=0.2,
        metric="minmax",
    )

    assert result.method == "medoid"
    assert result.fingerprint is None
    assert sum(result.weights.values()) == pytest.approx(1.0)
    assert len(result.neighbors) == 1
    selected = result.neighbors.iloc[0]
    assert selected["compound_id"] == 2
    assert selected["retrieval"] == "weighted_medoid"
    assert selected["source_spectrum_id"] == 20


def test_medoid_rejects_misaligned_pairwise_similarity(monkeypatch):
    search = make_search()
    monkeypatch.setattr(
        chemical_module,
        "pairwise_similarity",
        lambda fingerprints, metric: np.ones((1, 1)),
    )

    with pytest.raises(ContractError, match="changed candidate alignment"):
        search.aggregate_candidates(
            candidate_rows(),
            stored_records(),
            method="medoid",
            weight_source="rf_quality",
            top_n=3,
        )


def test_centroid_aggregation_builds_weighted_fingerprint_and_calls_search(monkeypatch):
    search = make_search()
    calls = []

    def fake_search(fingerprint, **kwargs):
        calls.append((np.asarray(fingerprint).copy(), kwargs))
        return pd.DataFrame([{"compound_id": 99, "chemical_rank": 1}])

    monkeypatch.setattr(search, "search", fake_search)

    result = search.aggregate_candidates(
        candidate_rows(),
        stored_records(),
        method="centroid",
        weight_source="rf_quality",
        weight_power=2.0,
        selection="top_n",
        top_n=2,
        score_delta=0.2,
        metric="minmax",
        pool=50,
        exact=False,
    )

    raw = np.asarray([0.8**2, 0.6**2], dtype=float)
    raw /= raw.sum()
    expected = raw[0] * np.asarray([1.0, 0.0]) + raw[1] * np.asarray([0.0, 1.0])

    assert result.method == "centroid"
    assert result.fingerprint.dtype == np.float32
    np.testing.assert_allclose(result.fingerprint, expected)
    assert len(calls) == 1
    fingerprint, kwargs = calls[0]
    np.testing.assert_allclose(fingerprint, expected)
    assert kwargs == {
        "top_n": 2,
        "pool": 50,
        "metric": "minmax",
        "exact": False,
    }


def test_empty_aggregation_returns_typed_empty_result():
    search = make_search()

    result = search.aggregate_candidates(
        [{"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.0}],
        stored_records(),
        method="medoid",
        weight_source="rf_quality",
    )

    assert result.is_empty
    assert result.weights == {}
    assert result.fingerprint is None
    assert result.neighbors.empty

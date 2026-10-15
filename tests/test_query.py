from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import ms2query_db.query as query_module
from ms2query_db.errors import ContractError
from ms2query_db.query import QueryEngine
from ms2query_db.results import SpectrumQueryResult


class DummyChemicalSearch:
    def __init__(self, database, projector):
        self.aggregate_calls = []
        self.next_aggregation = SimpleNamespace(
            method="medoid",
            weights={1: 1.0},
            fingerprint=None,
            neighbors=pd.DataFrame([{"compound_id": 1, "chemical_rank": 1}]),
            is_empty=False,
        )

    def aggregate_candidates(self, candidates, records, **kwargs):
        self.aggregate_calls.append((candidates, records, kwargs))
        return self.next_aggregation


class DummyFingerprinter:
    def compute_many(self, smiles):
        return [np.ones(4) for _ in smiles]


class DummyNative:
    def __init__(self):
        self.score_calls = []

    def search(self, collection, method, top_n, **kwargs):
        raise AssertionError("Unexpected native search")

    def score_candidates(self, query_collection, candidates, method):
        self.score_calls.append((query_collection, candidates, method))
        return [(0.73, 4)] * len(candidates)


class DummyReranker:
    def __init__(self, values=(0.5,)):
        self.values = values
        self.frames = []

    def predict(self, frame):
        self.frames.append(frame.copy())
        values = np.asarray(self.values, dtype=float)
        if len(values) == 1 and len(frame) != 1:
            values = np.repeat(values, len(frame))
        return values


def make_engine(
    monkeypatch,
    query_config_factory,
    *,
    library_mode="positive",
    config_overrides=None,
    embedder=SimpleNamespace(),
    native=None,
    reranker=None,
):
    monkeypatch.setattr(query_module, "ChemicalSearch", DummyChemicalSearch)
    config = query_config_factory(**(config_overrides or {}))
    database = SimpleNamespace(
        config=SimpleNamespace(
            ionmode=library_mode,
            query_processing=SimpleNamespace(),
            mz_precision=1e-6,
        )
    )
    return QueryEngine(
        database,
        embedder,
        DummyFingerprinter(),
        native or DummyNative(),
        projector=object(),
        config=config,
        reranker=reranker,
    )


def test_constructor_rejects_cross_mode_until_implemented(
    monkeypatch, query_config_factory
):
    monkeypatch.setattr(query_module, "ChemicalSearch", DummyChemicalSearch)
    config = query_config_factory(ionmode_matching="cross_mode")
    database = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(NotImplementedError, match="Cross-mode"):
        QueryEngine(
            database,
            SimpleNamespace(),
            DummyFingerprinter(),
            DummyNative(),
            object(),
            config,
        )


def test_constructor_requires_embedder_when_ms2deepscore_is_scored(
    monkeypatch, query_config_factory
):
    monkeypatch.setattr(query_module, "ChemicalSearch", DummyChemicalSearch)
    config = query_config_factory()
    database = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(ValueError, match="MS2DeepScore"):
        QueryEngine(
            database,
            None,
            DummyFingerprinter(),
            DummyNative(),
            object(),
            config,
        )


def test_constructor_requires_reranker_for_rf_weighted_aggregation(
    monkeypatch, query_config_factory
):
    monkeypatch.setattr(query_module, "ChemicalSearch", DummyChemicalSearch)
    config = query_config_factory(aggregation_method="medoid")
    database = SimpleNamespace(config=SimpleNamespace())

    with pytest.raises(ValueError, match="requires a candidate reranker"):
        QueryEngine(
            database,
            SimpleNamespace(),
            DummyFingerprinter(),
            DummyNative(),
            object(),
            config,
        )


class FakeCollection:
    def __init__(self, values):
        self.values = list(values)

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return FakeCollection(self.values[index])
        if isinstance(index, (int, np.integer)):
            return self.values[int(index)]
        indices = np.asarray(index, dtype=int).tolist()
        return FakeCollection([self.values[i] for i in indices])


def test_search_method_splits_mixed_query_collection_by_ionmode(
    monkeypatch, query_config_factory
):
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        library_mode="both",
    )
    calls = []

    def fake_backend(collection, embeddings, method, *, ionmode=None):
        calls.append((collection.values, embeddings, method, ionmode))
        return [[{"mode": ionmode, "query": value}] for value in collection.values]

    monkeypatch.setattr(engine, "_search_backend", fake_backend)

    result = engine._search_method(
        FakeCollection(["q0", "q1", "q2"]),
        ["e0", "e1", "e2"],
        ("positive", "negative", "positive"),
        "ms2deepscore",
    )

    assert calls == [
        (["q0", "q2"], ["e0", "e2"], "ms2deepscore", "positive"),
        (["q1"], ["e1"], "ms2deepscore", "negative"),
    ]
    assert result == [
        [{"mode": "positive", "query": "q0"}],
        [{"mode": "negative", "query": "q1"}],
        [{"mode": "positive", "query": "q2"}],
    ]


def test_candidate_rows_deduplicates_overlap_but_preserves_method_provenance(
    monkeypatch, query_config_factory
):
    engine = make_engine(monkeypatch, query_config_factory)
    engine.config.top_n = 2
    monkeypatch.setattr(
        query_module,
        "metadata_columns",
        lambda record: {
            "compound_id": record["compound_id"],
            "identity_key": record["identity_key"],
        },
    )
    stored = {
        10: {
            "compound_id": 1,
            "identity_key": "A",
            "precursor_mz": 101.0,
            "ionmode": "positive",
            "spectrum_hash": "h10",
        },
        20: {
            "compound_id": 2,
            "identity_key": "B",
            "precursor_mz": 102.0,
            "ionmode": "positive",
            "spectrum_hash": "h20",
        },
        30: {
            "compound_id": 3,
            "identity_key": "C",
            "precursor_mz": 103.0,
            "ionmode": "positive",
            "spectrum_hash": "h30",
        },
    }
    retrievals = {
        "ms2deepscore": [[
            {"spectrum_id": 10, "score": 0.9},
            {"spectrum_id": 20, "score": 0.8},
        ]],
        "entropy_hybrid": [[
            {"spectrum_id": 20, "score": 0.95, "matches": 6},
            {"spectrum_id": 30, "score": 0.7, "matches": 4},
        ]],
    }
    result = SpectrumQueryResult("q:0", 0)

    rows, hits = engine._candidate_rows(
        query_row=0,
        result=result,
        precursor=100.0,
        ionmode="positive",
        retrievals=retrievals,
        stored=stored,
    )

    assert [row["spectrum_id"] for row in rows] == [10, 20, 30]
    overlap = next(row for row in rows if row["spectrum_id"] == 20)
    assert overlap["retrieved_by"] == ["ms2deepscore", "entropy_hybrid"]
    assert overlap["rank_ms2deepscore"] == 2
    assert overlap["rank_entropy_hybrid"] == 1
    assert overlap["score_ms2deepscore"] == pytest.approx(0.8)
    assert overlap["score_entropy_hybrid"] == pytest.approx(0.95)
    assert len(hits) == 4


def test_candidate_rows_rejects_wrong_mode_from_same_mode_backend(
    monkeypatch, query_config_factory
):
    engine = make_engine(monkeypatch, query_config_factory)
    engine.config.top_n = 1
    monkeypatch.setattr(
        query_module,
        "metadata_columns",
        lambda record: {"compound_id": record["compound_id"]},
    )
    stored = {
        10: {
            "compound_id": 1,
            "precursor_mz": 101.0,
            "ionmode": "negative",
            "spectrum_hash": "h10",
        }
    }

    with pytest.raises(ContractError, match="Same-mode search returned"):
        engine._candidate_rows(
            query_row=0,
            result=SpectrumQueryResult("q:0", 0),
            precursor=100.0,
            ionmode="positive",
            retrievals={"ms2deepscore": [[{"spectrum_id": 10, "score": 0.9}]]},
            stored=stored,
        )


def test_candidate_rows_rejects_duplicate_id_within_one_method(
    monkeypatch, query_config_factory
):
    engine = make_engine(monkeypatch, query_config_factory)
    engine.config.top_n = 2
    monkeypatch.setattr(query_module, "metadata_columns", lambda record: {})
    stored = {
        10: {
            "precursor_mz": 101.0,
            "ionmode": "positive",
            "spectrum_hash": "h10",
        }
    }

    with pytest.raises(ContractError, match="duplicate spectrum IDs"):
        engine._candidate_rows(
            query_row=0,
            result=SpectrumQueryResult("q:0", 0),
            precursor=100.0,
            ionmode="positive",
            retrievals={
                "ms2deepscore": [[
                    {"spectrum_id": 10, "score": 0.9},
                    {"spectrum_id": 10, "score": 0.8},
                ]]
            },
            stored=stored,
        )


def test_score_union_fills_complementary_scores_without_changing_ranks(
    monkeypatch, query_config_factory
):
    native = DummyNative()
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        native=native,
    )
    rows = [
        {
            "spectrum_id": 10,
            "score_ms2deepscore": 0.9,
            "rank_ms2deepscore": 1,
            "score_entropy_hybrid": None,
            "rank_entropy_hybrid": None,
        },
        {
            "spectrum_id": 20,
            "score_ms2deepscore": None,
            "rank_ms2deepscore": None,
            "score_entropy_hybrid": 0.95,
            "rank_entropy_hybrid": 1,
        },
    ]
    stored = {
        10: {"embedding": np.asarray([1.0, 0.0])},
        20: {"embedding": np.asarray([0.6, 0.8])},
    }
    query_collection = FakeCollection(["query"])
    sentinel_candidates = FakeCollection(["candidate"])
    monkeypatch.setattr(
        query_module,
        "collection_from_records",
        lambda records, mz_precision: sentinel_candidates,
    )

    engine._score_union(
        query_collection=query_collection,
        query_embedding=np.asarray([1.0, 0.0]),
        rows=rows,
        stored=stored,
    )

    assert rows[0]["score_entropy_hybrid"] == pytest.approx(0.73)
    assert rows[0]["matches_entropy_hybrid"] == 4
    assert rows[0]["rank_entropy_hybrid"] is None
    assert rows[1]["score_ms2deepscore"] == pytest.approx(0.6)
    assert rows[1]["rank_ms2deepscore"] is None
    assert native.score_calls == [
        (query_collection, sentinel_candidates, "entropy_hybrid")
    ]


def test_apply_reranker_clips_quality_to_unit_interval(
    monkeypatch, query_config_factory
):
    reranker = DummyReranker(values=(-0.25, 1.4))
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        reranker=reranker,
    )
    monkeypatch.setattr(
        engine,
        "_candidate_frame",
        lambda rows: pd.DataFrame(rows),
    )
    rows = [{"spectrum_id": 1}, {"spectrum_id": 2}]

    engine._apply_reranker(rows)

    assert [row["rf_quality"] for row in rows] == [0.0, 1.0]


@pytest.mark.parametrize("values", [(0.5, 0.6, 0.7), (0.2, np.nan)])
def test_apply_reranker_rejects_invalid_output(
    monkeypatch, query_config_factory, values
):
    reranker = DummyReranker(values=values)
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        reranker=reranker,
    )
    monkeypatch.setattr(
        engine,
        "_candidate_frame",
        lambda rows: pd.DataFrame(rows),
    )
    rows = [{"spectrum_id": 1}, {"spectrum_id": 2}]

    with pytest.raises(ContractError):
        engine._apply_reranker(rows)


def test_apply_aggregation_delegates_to_chemical_service_and_sets_result(
    monkeypatch, query_config_factory
):
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        config_overrides={
            "aggregation_method": "medoid",
            "aggregation_weight_source": "rf_quality",
            "aggregation_weight_power": 2.0,
            "aggregation_selection": "score_delta",
            "aggregation_top_n": 20,
            "aggregation_score_delta": 0.2,
            "chemical_search_pool": 500,
        },
        reranker=DummyReranker(),
    )
    rows = [{"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.8}]
    stored = {10: {"compound_id": 1, "fingerprint": np.asarray([1.0, 0.0])}}
    result = SpectrumQueryResult("q:0", 0)

    engine._apply_aggregation(result, rows, stored)

    assert len(engine.chemical.aggregate_calls) == 1
    called_rows, called_stored, kwargs = engine.chemical.aggregate_calls[0]
    assert called_rows is rows
    assert called_stored is stored
    assert kwargs == {
        "method": "medoid",
        "weight_source": "rf_quality",
        "weight_power": 2.0,
        "selection": "score_delta",
        "top_n": 20,
        "score_delta": 0.2,
        "metric": "minmax",
        "pool": 500,
        "exact": False,
    }
    assert result.aggregation_method == "medoid"
    assert result.aggregation_weights == {1: 1.0}
    assert result.aggregation_fingerprint is None
    assert result.aggregation_results.iloc[0]["compound_id"] == 1


def test_apply_aggregation_warns_on_empty_candidate_support(
    monkeypatch, query_config_factory
):
    engine = make_engine(
        monkeypatch,
        query_config_factory,
        config_overrides={"aggregation_method": "medoid"},
        reranker=DummyReranker(),
    )
    engine.chemical.next_aggregation = SimpleNamespace(
        method="medoid",
        weights={},
        fingerprint=None,
        neighbors=pd.DataFrame(),
        is_empty=True,
    )
    result = SpectrumQueryResult("q:0", 0)

    engine._apply_aggregation(
        result,
        [{"spectrum_id": 10, "compound_id": 1, "rf_quality": 0.0}],
        {10: {"compound_id": 1, "fingerprint": np.asarray([1.0, 0.0])}},
    )

    assert result.aggregation_method is None
    assert result.aggregation_weights == {}
    assert any("positive candidate weights" in warning for warning in result.warnings)

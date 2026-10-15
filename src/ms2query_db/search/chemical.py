"""Chemical nearest-neighbour search and candidate aggregation.

Spectral retrieval and chemical aggregation are deliberately separate. Spectral
methods produce candidate spectra. This module can then rerank chemical-library
neighbors for a fingerprint query or combine retrieved candidate compounds via a
weighted medoid or centroid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..chemistry import pairwise_similarity, similarity_to_many
from ..errors import ContractError

CHEMICAL_HIT_COLUMNS = (
    "compound_id",
    "identity_key",
    "inchikey14",
    "inchikey",
    "smiles",
    "inchi",
    "formula",
    "exact_mass",
    "superclass",
    "class",
    "subclass",
    "visual_x",
    "visual_y",
    "pubchem_cid",
    "chemical_similarity",
    "chemical_rank",
    "retrieval",
)


def metadata_columns(record: Mapping) -> dict:
    """Return public compound metadata without allowing source-key collisions.

    Known columns use stable names. Arbitrary source metadata are prefixed with
    ``compound_meta__`` so imported fields can never overwrite candidate IDs,
    scores, ranks, or other MS2Query-DB result columns.
    """
    keys = (
        "compound_id",
        "identity_key",
        "inchikey14",
        "inchikey",
        "smiles",
        "inchi",
        "formula",
        "exact_mass",
        "superclass",
        "subclass",
        "visual_x",
        "visual_y",
        "pubchem_cid",
    )
    row = {
        key: record.get(key)
        for key in keys
    }
    row["class"] = record.get("compound_class")
    row.update(
        {
            f"compound_meta__{key}": value
            for key, value in (record.get("extra_metadata") or {}).items()
        }
    )
    return row


@dataclass(frozen=True)
class CandidateAggregationResult:
    """Result of weighted candidate-compound aggregation.

    Attributes
    ----------
    method
        ``"medoid"`` or ``"centroid"``.
    weights
        Normalized compound-level support weights after replicate aggregation,
        candidate selection, and the configured power transformation.
    fingerprint
        Weighted centroid fingerprint for centroid aggregation, otherwise
        ``None``.
    neighbors
        One-row medoid result or chemical-database neighbors for a centroid.
    """

    method: str
    weights: dict[int, float]
    fingerprint: np.ndarray | None
    neighbors: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        """Whether no positively weighted candidate compound was available."""
        return not self.weights


@dataclass(frozen=True)
class _CompoundEntry:
    compound_id: int
    score: float
    spectrum_id: int
    fingerprint: np.ndarray
    record: Mapping


class ChemicalSearch:
    """Exact or projected-ANN search over stored compound fingerprints."""

    def __init__(self, database, projector):
        self.database = database
        self.projector = projector

    def _validate_fingerprint(self, fingerprint) -> np.ndarray:
        query = np.asarray(
            fingerprint,
            dtype=np.float32,
        )
        expected = (
            self.database.config.fingerprints.size,
        )
        if query.shape != expected:
            raise ValueError(
                f"Chemical fingerprint must have shape {expected}; got {query.shape}."
            )
        if not np.isfinite(query).all():
            raise ValueError(
                "Chemical fingerprints must contain only finite values."
            )
        if np.any(query < 0):
            raise ValueError(
                "Chemical fingerprints must be nonnegative."
            )
        if not np.any(query > 0):
            raise ValueError(
                "Chemical fingerprints must contain at least one positive value."
            )
        return query

    def search(
        self,
        fingerprint,
        *,
        top_n: int = 20,
        pool: int = 500,
        metric: str = "minmax",
        exact: bool = False,
        exclude_inchikey14: str | None = None,
    ) -> pd.DataFrame:
        """Search the chemical library and exact-rerank returned fingerprints.

        Approximate search projects the query into the library's pgvector space,
        retrieves ``pool`` compounds, and reranks those compounds with the full
        stored fingerprint and requested chemical metric. Exact search scans the
        full compound library in keyset-paged blocks.
        """
        query = self._validate_fingerprint(fingerprint)

        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if isinstance(pool, bool) or not isinstance(pool, int) or pool < top_n:
            raise ValueError("pool must be an integer >= top_n.")

        if exact:
            blocks = self.database.iter_compounds()
            retrieval = "exact"
        else:
            projected = np.asarray(
                self.projector.transform(query[None, :])[0],
                dtype=np.float32,
            )
            records = self.database.chemical_candidates(
                projected,
                pool,
            )
            blocks = (records,)
            retrieval = "projected_ann_reranked"

        best: list[dict] = []
        for records in blocks:
            filtered = [
                record
                for record in records
                if (
                    exclude_inchikey14 is None
                    or record.get("inchikey14") != exclude_inchikey14
                )
            ]
            if not filtered:
                continue

            fingerprints = np.stack(
                [record["fingerprint"] for record in filtered]
            )
            scores = similarity_to_many(
                query,
                fingerprints,
                metric,
            )
            if len(scores) != len(filtered):
                raise ContractError(
                    "Chemical similarity calculation changed candidate alignment."
                )
            if not np.isfinite(scores).all():
                raise ContractError(
                    "Chemical similarity calculation produced nonfinite scores."
                )

            best.extend(
                {
                    **metadata_columns(record),
                    "chemical_similarity": float(score),
                    "retrieval": retrieval,
                }
                for record, score in zip(
                    filtered,
                    scores,
                    strict=True,
                )
            )
            best.sort(
                key=lambda row: (
                    -row["chemical_similarity"],
                    row["compound_id"],
                )
            )
            del best[top_n:]

        for rank, row in enumerate(best, 1):
            row["chemical_rank"] = rank

        if not best:
            return pd.DataFrame(columns=CHEMICAL_HIT_COLUMNS)

        columns = list(
            dict.fromkeys(
                CHEMICAL_HIT_COLUMNS
                + tuple(
                    key
                    for row in best
                    for key in row
                )
            )
        )
        return pd.DataFrame(best).reindex(columns=columns)

    @staticmethod
    def _compound_entries(
        candidates: Sequence[Mapping],
        records: Mapping[int, Mapping],
        *,
        weight_source: str,
        selection: str,
        top_n: int,
        score_delta: float,
        weight_power: float,
    ) -> list[tuple[_CompoundEntry, float]]:
        """Collapse replicate spectra and produce normalized compound weights.

        Replicate spectra use maximum support per compound. Selection operates on
        the untransformed support score; the positive power transformation is
        applied only after selecting compounds.
        """
        if weight_source == "rf_quality":
            score_column = "rf_quality"
        else:
            score_column = f"score_{weight_source}"

        compounds: dict[int, _CompoundEntry] = {}
        for row in candidates:
            score = row.get(score_column)
            if score is None:
                continue
            try:
                score = float(score)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(score) or score <= 0:
                continue

            spectrum_id = int(row["spectrum_id"])
            if spectrum_id not in records:
                raise ContractError(
                    f"Candidate spectrum {spectrum_id} has no stored record."
                )
            record = records[spectrum_id]
            compound_id = int(row["compound_id"])

            fingerprint = np.asarray(
                record["fingerprint"],
                dtype=np.float64,
            )
            if (
                fingerprint.ndim != 1
                or not fingerprint.size
                or not np.isfinite(fingerprint).all()
                or np.any(fingerprint < 0)
            ):
                raise ContractError(
                    f"Candidate compound {compound_id} has an invalid fingerprint."
                )

            previous = compounds.get(compound_id)
            if previous is None or score > previous.score:
                compounds[compound_id] = _CompoundEntry(
                    compound_id=compound_id,
                    score=score,
                    spectrum_id=spectrum_id,
                    fingerprint=fingerprint,
                    record=record,
                )

        entries = sorted(
            compounds.values(),
            key=lambda entry: (
                -entry.score,
                entry.compound_id,
            ),
        )
        if not entries:
            return []

        if selection == "top_n":
            entries = entries[:top_n]
        elif selection == "score_delta":
            cutoff = entries[0].score - score_delta
            entries = [
                entry
                for entry in entries
                if entry.score >= cutoff
            ]
        else:
            raise ValueError(
                "selection must be 'top_n' or 'score_delta'."
            )

        weights = np.asarray(
            [entry.score for entry in entries],
            dtype=np.float64,
        ) ** float(weight_power)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0:
            return []
        weights /= total

        return [
            (entry, float(weight))
            for entry, weight in zip(
                entries,
                weights,
                strict=True,
            )
        ]

    def aggregate_candidates(
        self,
        candidates: Sequence[Mapping],
        records: Mapping[int, Mapping],
        *,
        method: str,
        weight_source: str,
        weight_power: float = 2.0,
        selection: str = "top_n",
        top_n: int = 20,
        score_delta: float = 0.2,
        metric: str = "minmax",
        pool: int = 500,
        exact: bool = False,
    ) -> CandidateAggregationResult:
        """Aggregate retrieved candidates by weighted medoid or centroid.

        ``medoid`` selects a real retrieved compound maximizing RF/spectral
        weight-adjusted chemical centrality among the selected candidate
        compounds. ``centroid`` constructs a weighted fingerprint and searches
        the full chemical library from that synthetic fingerprint.
        """
        if method not in {"medoid", "centroid"}:
            raise ValueError("method must be 'medoid' or 'centroid'.")
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise ValueError("top_n must be a positive integer.")
        if not np.isfinite(weight_power) or weight_power <= 0:
            raise ValueError("weight_power must be finite and > 0.")
        if not np.isfinite(score_delta) or score_delta < 0:
            raise ValueError("score_delta must be finite and >= 0.")

        weighted = self._compound_entries(
            candidates,
            records,
            weight_source=weight_source,
            selection=selection,
            top_n=top_n,
            score_delta=score_delta,
            weight_power=weight_power,
        )
        if not weighted:
            return CandidateAggregationResult(
                method=method,
                weights={},
                fingerprint=None,
                neighbors=pd.DataFrame(columns=CHEMICAL_HIT_COLUMNS),
            )

        weights = {
            entry.compound_id: weight
            for entry, weight in weighted
        }

        if method == "centroid":
            fingerprint = np.sum(
                np.stack(
                    [
                        weight * entry.fingerprint
                        for entry, weight in weighted
                    ]
                ),
                axis=0,
            ).astype(np.float32)
            neighbors = self.search(
                fingerprint,
                top_n=top_n,
                pool=pool,
                metric=metric,
                exact=exact,
            )
            return CandidateAggregationResult(
                method=method,
                weights=weights,
                fingerprint=fingerprint,
                neighbors=neighbors,
            )

        fingerprints = [
            entry.fingerprint
            for entry, _ in weighted
        ]
        similarities = np.asarray(
            pairwise_similarity(
                fingerprints,
                metric,
            ),
            dtype=np.float64,
        )
        expected_shape = (len(weighted), len(weighted))
        if similarities.shape != expected_shape:
            raise ContractError(
                "Pairwise chemical similarity matrix changed candidate alignment."
            )
        if not np.isfinite(similarities).all():
            raise ContractError(
                "Pairwise chemical similarity produced nonfinite values."
            )

        weight_vector = np.asarray(
            [weight for _, weight in weighted],
            dtype=np.float64,
        )
        centrality = similarities @ weight_vector

        best = min(
            range(len(weighted)),
            key=lambda index: (
                -float(centrality[index]),
                -weighted[index][0].score,
                weighted[index][0].compound_id,
            ),
        )
        entry, selected_weight = weighted[best]

        row = {
            **metadata_columns(entry.record),
            "chemical_rank": 1,
            "retrieval": "weighted_medoid",
            "chemical_similarity": np.nan,
            "aggregation_score": float(centrality[best]),
            "aggregation_support": float(entry.score),
            "aggregation_weight": float(selected_weight),
            "source_spectrum_id": entry.spectrum_id,
        }
        columns = list(
            dict.fromkeys(
                CHEMICAL_HIT_COLUMNS
                + tuple(row)
            )
        )

        return CandidateAggregationResult(
            method=method,
            weights=weights,
            fingerprint=None,
            neighbors=pd.DataFrame([row]).reindex(columns=columns),
        )

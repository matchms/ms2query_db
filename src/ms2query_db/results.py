"""Public result objects and export helpers for MS2Query-DB queries.

The result model keeps four concepts separate:

* query status and diagnostics;
* method-specific retrieval hits;
* the deduplicated candidate-spectrum union;
* optional candidate aggregation (weighted medoid or fingerprint centroid).

Retrieval rank is method-specific and must never be interpreted as a fused rank.
Likewise, raw scores from different spectral methods are not directly comparable.
"""
from __future__ import annotations

import json
import os
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import METHODS
from .util import atomic_json, jsonable

RESULT_FORMAT_VERSION = 1
QUERY_STATUSES = ("pending", "ok", "no_hits", "rejected")
AGGREGATION_METHODS = ("medoid", "centroid")


CANDIDATE_COLUMNS = (
    "query_id",
    "query_index",
    "query_ionmode",
    "spectrum_id",
    "spectrum_hash",
    "candidate_ionmode",
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
    "query_precursor_mz",
    "candidate_precursor_mz",
    "precursor_mz_difference",
    "abs_precursor_mz_difference",
    "retrieved_by",
    "n_retrieval_methods",
    *(
        f"{prefix}_{method}"
        for method in METHODS
        for prefix in ("rank", "score", "matches")
    ),
    "rf_quality",
)

HIT_COLUMNS = (
    "query_id",
    "query_index",
    "spectrum_id",
    "search_metric",
    "search_rank",
    "search_score",
    "search_matches",
)

QUERY_COLUMNS = (
    "query_id",
    "query_index",
    "status",
    "reason_code",
    "reason",
    "query_ionmode",
    "query_precursor_mz",
    "n_candidates",
    "n_hits",
    "aggregation_method",
    "n_aggregation_compounds",
    "n_aggregation_results",
    "warnings",
    "query_metadata",
    "timings",
)

AGGREGATION_WEIGHT_COLUMNS = (
    "query_id",
    "query_index",
    "aggregation_method",
    "compound_id",
    "weight",
)


def _empty_frame(columns) -> pd.DataFrame:
    """Return an empty DataFrame with stable column order."""
    return pd.DataFrame(columns=list(columns))


def _stable_columns(base_columns, frames_or_rows) -> list[str]:
    """Return stable base columns followed by first-seen extension columns."""
    columns = list(base_columns)
    seen = set(columns)

    if isinstance(frames_or_rows, pd.DataFrame):
        iterable = frames_or_rows.columns
    else:
        iterable = (
            key
            for row in frames_or_rows
            for key in row
        )

    for key in iterable:
        if key not in seen:
            columns.append(key)
            seen.add(key)

    return columns


def _json_cell(value):
    """Serialize nested values deterministically for tabular CSV export."""
    if isinstance(value, np.ndarray):
        value = value.tolist()

    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(
            jsonable(value),
            sort_keys=True,
            separators=(",", ":"),
        )

    return value


def _csv_safe(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose nested values are JSON strings suitable for CSV."""
    frame = frame.copy()
    for column in frame.columns:
        if frame[column].map(
            lambda value: isinstance(
                value,
                (dict, list, tuple, set, np.ndarray),
            )
        ).any():
            frame[column] = frame[column].map(_json_cell)
    return frame


def _atomic_csv_gz(frame: pd.DataFrame, path: Path) -> None:
    """Write one gzip-compressed CSV and atomically publish the completed file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )
    try:
        _csv_safe(frame).to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "compresslevel": 1},
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


@dataclass
class SpectrumQueryResult:
    """Result for one input query spectrum.

    Parameters
    ----------
    query_id
        Stable query identifier for this run.
    query_index
        Source-row index used to preserve input/output alignment.

    Notes
    -----
    ``hits`` contains one row per retrieval method and rank. ``candidates`` is
    deduplicated by reference spectrum ID, so a spectrum retrieved by several
    methods occurs only once there while retaining its method-specific ranks,
    scores, and ``retrieved_by`` provenance.

    Candidate aggregation is optional. A medoid returns a real retrieved
    compound; a centroid can return chemical-library neighbors and an explicit
    synthetic fingerprint. Aggregation state is stored independently from the
    spectral candidate table.
    """

    query_id: str
    query_index: int

    status: str = "pending"
    reason_code: str | None = None
    reason: str | None = None

    query_ionmode: str | None = None
    query_precursor_mz: float | None = None
    query_metadata: dict = field(default_factory=dict)

    candidates: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(CANDIDATE_COLUMNS)
    )
    hits: pd.DataFrame = field(
        default_factory=lambda: _empty_frame(HIT_COLUMNS)
    )

    # Evaluation-only diagnostic when query structure is known.
    compound_similarity: pd.DataFrame | None = None

    # Optional candidate aggregation result.
    aggregation_method: str | None = None
    aggregation_weights: dict[int, float] = field(default_factory=dict)
    aggregation_fingerprint: np.ndarray | None = None
    aggregation_results: pd.DataFrame = field(default_factory=pd.DataFrame)

    warnings: list[str] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id:
            raise ValueError("query_id must be a non-empty string.")
        if (
            isinstance(self.query_index, bool)
            or not isinstance(self.query_index, int)
            or self.query_index < 0
        ):
            raise ValueError("query_index must be a nonnegative integer.")

    @property
    def has_aggregation(self) -> bool:
        """Whether a medoid/centroid aggregation result is attached."""
        return self.aggregation_method is not None

    def set_aggregation(
        self,
        *,
        method: str,
        weights: Mapping[int, float],
        fingerprint: np.ndarray | None,
        results: pd.DataFrame,
    ) -> None:
        """Attach one validated candidate-aggregation result.

        Parameters
        ----------
        method
            ``"medoid"`` or ``"centroid"``.
        weights
            Normalized positive compound-level weights used for aggregation.
        fingerprint
            Synthetic centroid fingerprint. Must be ``None`` for a medoid.
        results
            One-row medoid result or chemical-library neighbors for a centroid.
        """
        if method not in AGGREGATION_METHODS:
            raise ValueError(
                f"aggregation method must be one of {AGGREGATION_METHODS}."
            )

        normalized_weights: dict[int, float] = {}
        for compound_id, weight in weights.items():
            if (
                isinstance(compound_id, bool)
                or not isinstance(compound_id, (int, np.integer))
            ):
                raise TypeError("Aggregation compound IDs must be integers.")
            value = float(weight)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    "Aggregation weights must be finite and strictly positive."
                )
            normalized_weights[int(compound_id)] = value

        if normalized_weights:
            total = float(sum(normalized_weights.values()))
            if not np.isclose(total, 1.0, rtol=1e-6, atol=1e-8):
                raise ValueError(
                    f"Aggregation weights must sum to 1; got {total:.12g}."
                )

        if method == "medoid" and fingerprint is not None:
            raise ValueError("Medoid aggregation must not store a fingerprint.")

        validated_fingerprint = None
        if fingerprint is not None:
            validated_fingerprint = np.asarray(
                fingerprint,
                dtype=np.float32,
            ).copy()
            if (
                validated_fingerprint.ndim != 1
                or not validated_fingerprint.size
                or not np.isfinite(validated_fingerprint).all()
                or np.any(validated_fingerprint < 0)
            ):
                raise ValueError(
                    "Aggregation fingerprints must be finite, nonnegative, "
                    "one-dimensional, and nonempty."
                )

        if not isinstance(results, pd.DataFrame):
            raise TypeError("Aggregation results must be a pandas DataFrame.")

        self.aggregation_method = method
        self.aggregation_weights = normalized_weights
        self.aggregation_fingerprint = validated_fingerprint
        self.aggregation_results = results.copy()

    def candidates_long(self) -> pd.DataFrame:
        """Return retrieval provenance joined to unique candidate metadata.

        The returned table has one row per ``query x retrieval method x rank``.
        A candidate retrieved by multiple methods therefore occurs once for each
        originating method. Raw scores remain method-specific and should not be
        compared directly across metrics.
        """
        if self.hits.empty:
            return _empty_frame(
                _stable_columns(
                    HIT_COLUMNS,
                    self.candidates,
                )
            )

        return self.hits.merge(
            self.candidates,
            on=["query_id", "query_index", "spectrum_id"],
            how="left",
            validate="many_to_one",
        )

    def aggregation_weights_dataframe(self) -> pd.DataFrame:
        """Return one row per compound weight used by candidate aggregation."""
        if not self.aggregation_weights:
            return _empty_frame(AGGREGATION_WEIGHT_COLUMNS)

        return pd.DataFrame(
            [
                {
                    "query_id": self.query_id,
                    "query_index": self.query_index,
                    "aggregation_method": self.aggregation_method,
                    "compound_id": compound_id,
                    "weight": weight,
                }
                for compound_id, weight in sorted(
                    self.aggregation_weights.items()
                )
            ],
            columns=AGGREGATION_WEIGHT_COLUMNS,
        )

    def aggregation_results_dataframe(self) -> pd.DataFrame:
        """Return medoid/centroid output annotated with its originating query."""
        if self.aggregation_results.empty:
            return pd.DataFrame()

        frame = self.aggregation_results.copy()
        frame.insert(0, "aggregation_method", self.aggregation_method)
        frame.insert(0, "query_index", self.query_index)
        frame.insert(0, "query_id", self.query_id)
        return frame


@dataclass
class BatchQueryResult:
    """Collection of row-aligned query results plus run-level provenance."""

    results: list[SpectrumQueryResult]
    provenance: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        query_ids = [result.query_id for result in self.results]
        query_indices = [result.query_index for result in self.results]
        if len(set(query_ids)) != len(query_ids):
            raise ValueError("Batch query IDs must be unique.")
        if len(set(query_indices)) != len(query_indices):
            raise ValueError("Batch query indices must be unique.")

    def candidates_dataframe(self) -> pd.DataFrame:
        """Concatenate unique candidate-spectrum tables across the batch."""
        frames = [
            result.candidates
            for result in self.results
            if not result.candidates.empty
        ]
        if not frames:
            return _empty_frame(CANDIDATE_COLUMNS)

        frame = pd.concat(frames, ignore_index=True, sort=False)
        return frame.reindex(
            columns=_stable_columns(CANDIDATE_COLUMNS, frame)
        )

    def hits_dataframe(self, *, include_metadata: bool = False) -> pd.DataFrame:
        """Concatenate retrieval-hit rows, optionally joined to candidate data."""
        frames = [
            (
                result.candidates_long()
                if include_metadata
                else result.hits
            )
            for result in self.results
            if not result.hits.empty
        ]
        if not frames:
            return _empty_frame(HIT_COLUMNS)
        return pd.concat(frames, ignore_index=True, sort=False)

    def queries_dataframe(self) -> pd.DataFrame:
        """Return one status/diagnostic row per input query."""
        rows = []
        for result in self.results:
            rows.append(
                {
                    "query_id": result.query_id,
                    "query_index": result.query_index,
                    "status": result.status,
                    "reason_code": result.reason_code,
                    "reason": result.reason,
                    "query_ionmode": result.query_ionmode,
                    "query_precursor_mz": result.query_precursor_mz,
                    "n_candidates": len(result.candidates),
                    "n_hits": len(result.hits),
                    "aggregation_method": result.aggregation_method,
                    "n_aggregation_compounds": len(result.aggregation_weights),
                    "n_aggregation_results": len(result.aggregation_results),
                    "warnings": list(result.warnings),
                    "query_metadata": dict(result.query_metadata),
                    "timings": dict(result.timings),
                }
            )
        return pd.DataFrame(rows, columns=QUERY_COLUMNS)

    def aggregation_weights_dataframe(self) -> pd.DataFrame:
        """Return compound-level aggregation weights for the entire batch."""
        frames = [
            result.aggregation_weights_dataframe()
            for result in self.results
            if result.aggregation_weights
        ]
        if not frames:
            return _empty_frame(AGGREGATION_WEIGHT_COLUMNS)
        return pd.concat(frames, ignore_index=True)

    def aggregation_results_dataframe(self) -> pd.DataFrame:
        """Return medoid/centroid outputs for the entire batch."""
        frames = [
            result.aggregation_results_dataframe()
            for result in self.results
            if not result.aggregation_results.empty
        ]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True, sort=False)

    def export(self, directory: str | Path) -> None:
        """Atomically export this in-memory query batch.

        The export is intentionally tabular. Nested query metadata, warning
        lists, timing dictionaries, and retrieval provenance are JSON-encoded in
        CSV cells. Compound-level aggregation weights receive their own table.
        Synthetic centroid fingerprints are stored separately as NumPy arrays.

        The command-line interface can call this method once per processed input
        batch; it does not assume that an entire large query run fits in memory.
        """
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)

        tables = {
            "queries": self.queries_dataframe(),
            "candidates": self.candidates_dataframe(),
            "hits": self.hits_dataframe(include_metadata=True),
            "aggregation_weights": self.aggregation_weights_dataframe(),
            "aggregation_results": self.aggregation_results_dataframe(),
        }

        for name, frame in tables.items():
            _atomic_csv_gz(
                frame,
                directory / f"{name}.csv.gz",
            )

        fingerprint_directory = directory / "aggregation_fingerprints"
        for result in self.results:
            if result.aggregation_fingerprint is None:
                continue
            fingerprint_directory.mkdir(exist_ok=True)
            target = fingerprint_directory / f"query_{result.query_index}.npy"
            temporary = target.with_name(
                f".{target.name}.{uuid.uuid4().hex}.tmp"
            )
            try:
                with temporary.open("wb") as handle:
                    np.save(
                        handle,
                        result.aggregation_fingerprint,
                        allow_pickle=False,
                    )
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()

        similarity_directory = directory / "compound_similarity"
        for result in self.results:
            if result.compound_similarity is None:
                continue
            similarity_directory.mkdir(exist_ok=True)
            _atomic_csv_gz(
                result.compound_similarity,
                similarity_directory
                / f"query_{result.query_index}.csv.gz",
            )

        atomic_json(
            directory / "provenance.json",
            self.provenance,
        )

        status_counts = (
            self.queries_dataframe()["status"]
            .value_counts(dropna=False)
            .to_dict()
        )
        manifest = {
            "format_version": RESULT_FORMAT_VERSION,
            "n_queries": len(self.results),
            "status_counts": status_counts,
            "rows": {
                name: len(frame)
                for name, frame in tables.items()
            },
            "has_aggregation_fingerprints": any(
                result.aggregation_fingerprint is not None
                for result in self.results
            ),
            "has_compound_similarity": any(
                result.compound_similarity is not None
                for result in self.results
            ),
        }
        atomic_json(
            directory / "manifest.json",
            manifest,
        )

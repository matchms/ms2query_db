"""Collection-oriented query orchestration for MS2Query-DB.

The query engine is responsible for orchestration, alignment, candidate provenance, optional
reranking, and candidate aggregation. Spectral algorithms themselves remain in backend
adapters, and chemical nearest-neighbour search remains in the chemical search component.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from .chemistry import identify_compound, pairwise_similarity
from .config import METHODS
from .embeddings import validated_embeddings
from .errors import ContractError, RejectedRecord
from .processing import collection_from_records, process_query_collection
from .results import CANDIDATE_COLUMNS, HIT_COLUMNS, SpectrumQueryResult
from .search.chemical import ChemicalSearch, metadata_columns
from .util import jsonable, text_value


class CandidateReranker(Protocol):
    """Protocol for learned candidate-quality rerankers.

    ``predict()`` receives the row-aligned candidate table for one *query
    batch*. The table can therefore contain several ``query_index`` groups. A
    reranker that uses query-relative features must compute those features
    independently within each query group, but it should call the underlying
    model once for the complete batch whenever possible.

    The returned array must contain one finite quality estimate per input row in
    exactly the same order. The current RF models predict expected chemical
    similarity in the interval [0, 1].
    """

    def predict(self, candidates: pd.DataFrame) -> np.ndarray: ...


class QueryEngine:
    """Execute batched spectral-library queries while preserving row alignment.

    Query spectra are processed as :class:`matchms.SpectraCollection` batches.
    Each configured retrieval method contributes its own top-N list; those lists
    are deduplicated by reference spectrum ID into one candidate union. Every
    method that is either used for retrieval or requested as an additional score
    is then evaluated on every candidate in that union, so learned rerankers see
    complete complementary spectral evidence.

    Parameters
    ----------
    database
        Open MS2Query-DB database adapter.
    embedder
        Library-compatible MS2DeepScore embedder, or ``None`` when embedding
        scores are not requested.
    fingerprinter
        Chemical fingerprinter used only to validate supplied query annotations.
        Query structures never become spectral-search or reranking features.
    native
        Classical matchms index store.
    projector
        Chemical fingerprint projector used by :class:`ChemicalSearch`.
    config
        :class:`~ms2query_db.config.QueryConfig` controlling this query run.
    processor
        Optional custom matchms ``SpectraProcessor``. It must expose
        ``process_collection()``.
    reranker
        Optional learned candidate reranker. Required when RF quality is used as
        an aggregation weight source.

    Notes
    -----
    ``ionmode_matching='cross_mode'`` is intentionally represented in the
    configuration API but is not implemented yet. Same-mode querying of a
    ``BuildConfig(ionmode='both')`` library requires the embedding and native
    backends to accept an ``ionmode=...`` restriction; the query engine already
    routes batches accordingly.
    """

    def __init__(
        self,
        database,
        embedder,
        fingerprinter,
        native,
        projector,
        config,
        *,
        processor=None,
        reranker: CandidateReranker | None = None,
    ):
        self.database = database
        self.embedder = embedder
        self.fingerprinter = fingerprinter
        self.native = native
        self.config = config
        self.reranker = reranker
        self.processor = processor
        self.chemical = ChemicalSearch(database, projector)

        if config.ionmode_matching == "cross_mode":
            raise NotImplementedError(
                "Cross-mode spectral search is not implemented yet. "
                "Use ionmode_matching='same_mode'."
            )

        score_methods = set(config.search_methods) | set(
            config.additional_top_n_scores or ()
        )
        self.score_methods = tuple(
            method for method in METHODS if method in score_methods
        )
        self.need_embedding = "ms2deepscore" in score_methods

        if self.need_embedding and embedder is None:
            raise ValueError(
                "MS2DeepScore retrieval/scoring requires the library's "
                "embedding model."
            )

        if (
            config.aggregation_method is not None
            and config.aggregation_weight_source == "rf_quality"
            and reranker is None
        ):
            raise ValueError(
                "aggregation_weight_source='rf_quality' requires a candidate "
                "reranker."
            )

        if processor is not None and not callable(
            getattr(processor, "process_collection", None)
        ):
            raise TypeError(
                "processor must expose process_collection()."
            )

    @staticmethod
    def _reject(result: SpectrumQueryResult, error: RejectedRecord) -> None:
        """Mark one query result as rejected without affecting its batch peers."""
        result.status = "rejected"
        result.reason_code = error.code
        result.reason = str(error)

    def _validate_query_annotations(
        self,
        records,
        results: Sequence[SpectrumQueryResult],
    ) -> tuple[list, list[int]]:
        """Validate optional structural annotations and retain usable spectra.

        Structural annotations are validation/evaluation metadata only. They are
        never returned as model features and are stripped before spectral
        processing.
        """
        spectra: list = []
        positions: list[int] = []
        annotated: list[tuple[int, str]] = []

        for position, (record, result) in enumerate(
            zip(records, results, strict=True)
        ):
            if record.error is not None:
                self._reject(result, record.error)
                continue

            spectrum = record.value
            if spectrum is None:
                self._reject(
                    result,
                    RejectedRecord(
                        "empty_spectrum",
                        "Missing query spectrum",
                    ),
                )
                continue

            result.query_metadata = jsonable(
                dict(spectrum.metadata)
            )

            try:
                smiles = text_value(spectrum.get("smiles"))
                inchi = text_value(spectrum.get("inchi"))

                if (
                    self.config.annotation_policy == "require"
                    and not (smiles or inchi)
                ):
                    raise RejectedRecord(
                        "missing_structure",
                        "Annotated-query mode requires SMILES or InChI",
                    )

                if (
                    self.config.annotation_policy != "ignore"
                    and (smiles or inchi)
                ):
                    identity = identify_compound(
                        smiles,
                        inchi,
                        text_value(spectrum.get("inchikey")),
                    )
                    annotated.append(
                        (position, identity.smiles)
                    )

            except RejectedRecord as exc:
                self._reject(result, exc)
                continue

            spectra.append(spectrum)
            positions.append(position)

        if not annotated:
            return spectra, positions

        annotation_positions = [item[0] for item in annotated]
        annotation_smiles = [item[1] for item in annotated]
        fp_results = self.fingerprinter.compute_many(
            annotation_smiles
        )

        if len(fp_results) != len(annotation_positions):
            raise ContractError(
                "Fingerprinter changed query annotation alignment"
            )

        rejected_positions: set[int] = set()
        for position, value in zip(
            annotation_positions,
            fp_results,
            strict=True,
        ):
            if isinstance(value, RejectedRecord):
                self._reject(results[position], value)
                rejected_positions.add(position)

        if not rejected_positions:
            return spectra, positions

        retained_spectra: list = []
        retained_positions: list[int] = []
        for spectrum, position in zip(
            spectra,
            positions,
            strict=True,
        ):
            if position not in rejected_positions:
                retained_spectra.append(spectrum)
                retained_positions.append(position)

        return retained_spectra, retained_positions

    def _process_queries(
        self,
        spectra,
        positions,
        results,
    ):
        """Process a query batch as one ``SpectraCollection``."""
        started = time.perf_counter()
        processed = process_query_collection(
            spectra,
            self.database.config.query_processing,
            library_ionmode=self.database.config.ionmode,
            mz_precision=self.database.config.mz_precision,
            processor=self.processor,
        )
        elapsed = time.perf_counter() - started

        for position in positions:
            results[position].timings[
                "batch_processing_seconds"
            ] = elapsed

        for local_row, error in processed.rejected.items():
            self._reject(
                results[positions[local_row]],
                error,
            )

        retained_positions = [
            positions[local_row]
            for local_row in processed.input_rows
        ]

        return processed, retained_positions

    def _compute_embeddings(
        self,
        collection,
        positions,
        precursors,
        ionmodes,
        results,
    ):
        """Compute row-aligned embeddings and remove only failed query rows."""
        if not self.need_embedding:
            return (
                collection,
                positions,
                precursors,
                ionmodes,
                None,
            )

        values = validated_embeddings(
            self.embedder,
            collection,
        )

        keep = np.ones(len(collection), dtype=bool)
        embeddings: list[np.ndarray] = []

        for row_index, (position, value) in enumerate(
            zip(positions, values, strict=True)
        ):
            if isinstance(value, RejectedRecord):
                self._reject(results[position], value)
                keep[row_index] = False
            else:
                embeddings.append(value)

        if not np.any(keep):
            return None, [], np.empty(0), (), None

        if np.all(keep):
            return (
                collection,
                positions,
                precursors,
                ionmodes,
                embeddings,
            )

        retained_collection = collection.filter(keep)
        retained_positions = [
            position
            for position, accepted in zip(
                positions,
                keep,
                strict=True,
            )
            if accepted
        ]
        retained_precursors = precursors[keep]
        retained_ionmodes = tuple(
            mode
            for mode, accepted in zip(
                ionmodes,
                keep,
                strict=True,
            )
            if accepted
        )

        return (
            retained_collection,
            retained_positions,
            retained_precursors,
            retained_ionmodes,
            embeddings,
        )

    def _search_backend(
        self,
        collection,
        embeddings,
        method: str,
        *,
        ionmode: str | None = None,
    ):
        """Run one search backend on one mode-homogeneous query collection."""
        cfg = self.config

        if method == "ms2deepscore":
            kwargs = {
                "ef_search": cfg.ef_search,
                "exact": cfg.exact_embedding_search,
            }
            if ionmode is not None:
                kwargs["ionmode"] = ionmode
            return self.database.search_embeddings(
                embeddings,
                cfg.top_n,
                **kwargs,
            )

        kwargs = {
            "query_batch": cfg.classical_query_batch,
            "n_jobs": cfg.classical_n_jobs,
        }
        if ionmode is not None:
            kwargs["ionmode"] = ionmode

        return self.native.search(
            collection,
            method,
            cfg.top_n,
            **kwargs,
        )

    def _search_method(
        self,
        collection,
        embeddings,
        ionmodes,
        method: str,
    ):
        """Search one method, splitting a dual-mode library by query ion mode."""
        if self.database.config.ionmode != "both":
            return self._search_backend(
                collection,
                embeddings,
                method,
            )

        output = [None] * len(collection)
        ionmode_array = np.asarray(ionmodes, dtype=object)

        for mode in ("positive", "negative"):
            indices = np.flatnonzero(ionmode_array == mode)
            if not len(indices):
                continue

            subset = collection[indices]
            subset_embeddings = (
                [embeddings[index] for index in indices]
                if embeddings is not None
                else None
            )
            groups = self._search_backend(
                subset,
                subset_embeddings,
                method,
                ionmode=mode,
            )

            if len(groups) != len(indices):
                raise ContractError(
                    "Ion-mode search backend lost query row alignment"
                )

            for index, group in zip(
                indices,
                groups,
                strict=True,
            ):
                output[int(index)] = group

        if any(group is None for group in output):
            raise ContractError(
                "Ion-mode routing left an unresolved query row"
            )

        return output

    def _retrieve(
        self,
        collection,
        embeddings,
        ionmodes,
        positions,
        results,
    ) -> dict[str, list[list[dict]]]:
        """Run all configured retrieval methods on the processed query batch."""
        retrievals: dict[str, list[list[dict]]] = {}

        for method in self.config.search_methods:
            started = time.perf_counter()
            groups = self._search_method(
                collection,
                embeddings,
                ionmodes,
                method,
            )

            if len(groups) != len(collection):
                raise ContractError(
                    "Search backend lost query row alignment"
                )

            retrievals[method] = groups
            elapsed = time.perf_counter() - started

            for position in positions:
                results[position].timings[
                    f"batch_search_{method}_seconds"
                ] = elapsed

        return retrievals

    @staticmethod
    def _all_spectrum_ids(retrievals) -> list[int]:
        return sorted(
            {
                hit["spectrum_id"]
                for groups in retrievals.values()
                for group in groups
                for hit in group
            }
        )

    def _candidate_rows(
        self,
        *,
        query_row: int,
        result: SpectrumQueryResult,
        precursor: float,
        ionmode: str,
        retrievals,
        stored,
    ) -> tuple[list[dict], list[dict]]:
        """Merge method-specific retrievals into one provenance-preserving union."""
        by_id: dict[int, dict] = {}
        hits: list[dict] = []

        for method, groups in retrievals.items():
            group = groups[query_row]

            if len(group) < self.config.top_n:
                result.warnings.append(
                    f"{method}: {len(group)} of {self.config.top_n} "
                    "requested hits returned"
                )

            seen: set[int] = set()

            for rank, hit in enumerate(group, 1):
                sid = int(hit["spectrum_id"])
                if sid in seen:
                    raise ContractError(
                        "One retrieval method returned duplicate spectrum IDs"
                    )
                seen.add(sid)

                record = stored[sid]
                candidate_mode = text_value(
                    record.get("ionmode")
                ).lower()

                if (
                    self.config.ionmode_matching == "same_mode"
                    and candidate_mode != ionmode
                ):
                    raise ContractError(
                        "Same-mode search returned a candidate with ionmode "
                        f"{candidate_mode!r} for a {ionmode!r} query"
                    )

                if sid not in by_id:
                    candidate_precursor = float(
                        record["precursor_mz"]
                    )
                    delta = candidate_precursor - precursor
                    row = dict(
                        metadata_columns(record),
                        query_id=result.query_id,
                        query_index=result.query_index,
                        query_ionmode=ionmode,
                        spectrum_id=sid,
                        spectrum_hash=record["spectrum_hash"],
                        candidate_ionmode=candidate_mode,
                        query_precursor_mz=precursor,
                        candidate_precursor_mz=candidate_precursor,
                        precursor_mz_difference=delta,
                        abs_precursor_mz_difference=abs(delta),
                        retrieved_by=[],
                    )
                    for scoring_method in METHODS:
                        row.update(
                            {
                                f"rank_{scoring_method}": None,
                                f"score_{scoring_method}": None,
                                f"matches_{scoring_method}": None,
                            }
                        )
                    by_id[sid] = row

                row = by_id[sid]
                row["retrieved_by"].append(method)
                row[f"rank_{method}"] = rank
                row[f"score_{method}"] = float(hit["score"])
                row[f"matches_{method}"] = hit.get("matches")

                hits.append(
                    {
                        "query_id": result.query_id,
                        "query_index": result.query_index,
                        "spectrum_id": sid,
                        "search_metric": method,
                        "search_rank": rank,
                        "search_score": float(hit["score"]),
                        "search_matches": hit.get("matches"),
                    }
                )

        ids = sorted(by_id)
        return [by_id[sid] for sid in ids], hits

    def _score_union(
        self,
        *,
        query_collection,
        query_embedding,
        rows: list[dict],
        stored: Mapping[int, Mapping],
    ) -> None:
        """Fill complementary spectral scores for every candidate in the union.

        Retrieval rank stays ``None`` when a method did not retrieve a candidate;
        only the missing score/match fields are completed. This distinction is
        required by learned rerankers that use both spectral evidence and
        retrieval provenance.
        """
        if not rows:
            return

        for method in self.score_methods:
            missing = [
                row
                for row in rows
                if row[f"score_{method}"] is None
            ]
            if not missing:
                continue

            if method == "ms2deepscore":
                if query_embedding is None:
                    raise ContractError(
                        "MS2DeepScore union scoring requested without a "
                        "query embedding"
                    )

                refs = [
                    stored[row["spectrum_id"]].get("embedding")
                    for row in missing
                ]
                if any(value is None for value in refs):
                    raise ContractError(
                        "Candidate lacks an embedding required for "
                        "complementary MS2DeepScore scoring"
                    )

                scores = (
                    np.asarray(refs, dtype=np.float64)
                    @ np.asarray(query_embedding, dtype=np.float64)
                )
                additions = [
                    (float(np.clip(score, -1.0, 1.0)), None)
                    for score in scores
                ]
            else:
                candidate_collection = collection_from_records(
                    [
                        stored[row["spectrum_id"]]
                        for row in missing
                    ],
                    mz_precision=self.database.config.mz_precision,
                )
                additions = self.native.score_candidates(
                    query_collection,
                    candidate_collection,
                    method,
                )

            if len(additions) != len(missing):
                raise ContractError(
                    "Complementary-score result lost candidate alignment"
                )

            for row, (score, matches) in zip(
                missing,
                additions,
                strict=True,
            ):
                row[f"score_{method}"] = float(score)
                row[f"matches_{method}"] = matches

    def _apply_reranker_batch(
        self,
        row_groups: Sequence[list[dict]],
    ) -> None:
        """Attach reranker estimates with one model call per query batch.

        Candidate rows remain grouped by query for retrieval, complementary
        scoring, and aggregation. Only the learned inference step is flattened:
        every nonempty group is concatenated in stable query/candidate order,
        passed to the reranker once, and then written back to the same row
        objects. ``query_index`` remains present in the candidate frame so a
        reranker can build query-relative features independently per group.
        """
        if self.reranker is None:
            return

        rows = [
            row
            for group in row_groups
            for row in group
        ]
        if not rows:
            return

        frame = self._candidate_frame(rows)
        values = np.asarray(
            self.reranker.predict(frame),
            dtype=np.float64,
        )

        if values.shape != (len(rows),):
            raise ContractError(
                "Candidate reranker changed row alignment"
            )
        if not np.isfinite(values).all():
            raise ContractError(
                "Candidate reranker returned nonfinite values"
            )

        values = np.clip(values, 0.0, 1.0)
        for row, value in zip(rows, values, strict=True):
            row["rf_quality"] = float(value)

    def _apply_reranker(self, rows: list[dict]) -> None:
        """Rerank one candidate group; retained for focused callers/tests."""
        self._apply_reranker_batch([rows])

    @staticmethod
    def _candidate_frame(rows: list[dict]) -> pd.DataFrame:
        """Create the canonical candidate DataFrame without losing extra columns."""
        columns = list(
            dict.fromkeys(
                list(CANDIDATE_COLUMNS)
                + [key for row in rows for key in row]
            )
        )
        frame = pd.DataFrame(rows).reindex(columns=columns)

        for method in METHODS:
            rank = f"rank_{method}"
            matches = f"matches_{method}"
            if rank in frame:
                frame[rank] = frame[rank].astype("Int64")
            if matches in frame:
                frame[matches] = frame[matches].astype("Int64")

        return frame

    def _apply_aggregation(
        self,
        result: SpectrumQueryResult,
        rows: list[dict],
        stored,
    ) -> None:
        """Apply configured medoid or centroid aggregation to candidate rows.

        Chemical aggregation lives in :mod:`ms2query_db.search.chemical`; this
        method only transfers its typed result onto ``SpectrumQueryResult``.
        """
        cfg = self.config
        if cfg.aggregation_method is None or not rows:
            return

        aggregation = self.chemical.aggregate_candidates(
            rows,
            stored,
            method=cfg.aggregation_method,
            weight_source=cfg.aggregation_weight_source,
            weight_power=cfg.aggregation_weight_power,
            selection=cfg.aggregation_selection,
            top_n=cfg.aggregation_top_n,
            score_delta=cfg.aggregation_score_delta,
            metric=cfg.chemical_metric,
            pool=cfg.chemical_search_pool,
            exact=cfg.exact_chemical_search,
        )

        if aggregation.is_empty:
            result.warnings.append(
                "No positive candidate weights for aggregation"
            )
            return

        result.set_aggregation(
            method=aggregation.method,
            weights=aggregation.weights,
            fingerprint=aggregation.fingerprint,
            results=aggregation.neighbors,
        )

    def process_batch(
        self,
        records,
        *,
        source_label: str = "query",
    ) -> list[SpectrumQueryResult]:
        """Process and search one batch of input records.

        Rejections are row-local: invalid queries receive a rejected result while
        valid peers continue through batch processing, embedding, retrieval, and
        candidate aggregation.
        """
        records = list(records)
        results = [
            SpectrumQueryResult(
                f"{source_label}:{record.index}",
                record.index,
            )
            for record in records
        ]

        spectra, positions = self._validate_query_annotations(
            records,
            results,
        )
        if not spectra:
            return results

        processed, positions = self._process_queries(
            spectra,
            positions,
            results,
        )
        if processed.spectra is None:
            return results

        (
            collection,
            positions,
            precursors,
            ionmodes,
            embeddings,
        ) = self._compute_embeddings(
            processed.spectra,
            positions,
            processed.precursor_mz,
            processed.ionmodes,
            results,
        )
        if collection is None or not len(collection):
            return results

        retrievals = self._retrieve(
            collection,
            embeddings,
            ionmodes,
            positions,
            results,
        )

        all_ids = self._all_spectrum_ids(retrievals)
        stored = self.database.get_candidates(all_ids)
        if set(stored) != set(all_ids):
            raise ContractError(
                "Search backend returned unresolved spectrum IDs"
            )

        pending: list[
            tuple[SpectrumQueryResult, list[dict], list[dict], list[int]]
        ] = []

        # Build and fully score each query's candidate union first. Learned
        # reranking is deliberately deferred until all query groups are ready so
        # the model can evaluate the complete outer batch in one call.
        for query_row, position in enumerate(positions):
            result = results[position]
            result.query_ionmode = ionmodes[query_row]
            result.query_precursor_mz = float(precursors[query_row])
            rows, hits = self._candidate_rows(
                query_row=query_row,
                result=result,
                precursor=float(precursors[query_row]),
                ionmode=ionmodes[query_row],
                retrievals=retrievals,
                stored=stored,
            )

            ids = [row["spectrum_id"] for row in rows]
            query_embedding = (
                embeddings[query_row]
                if embeddings is not None
                else None
            )

            self._score_union(
                query_collection=collection[query_row : query_row + 1],
                query_embedding=query_embedding,
                rows=rows,
                stored=stored,
            )

            for row in rows:
                row["n_retrieval_methods"] = len(
                    row["retrieved_by"]
                )

            pending.append((result, rows, hits, ids))

        self._apply_reranker_batch(
            [rows for _, rows, _, _ in pending]
        )

        # Final result objects stay query-local. Aggregation is intentionally
        # performed after batched reranking because RF quality can be its weight source.
        for result, rows, hits, ids in pending:
            result.candidates = self._candidate_frame(rows)
            result.hits = pd.DataFrame(
                hits,
                columns=HIT_COLUMNS,
            )

            if self.config.compute_chemical_similarity and rows:
                fingerprints = {
                    stored[sid]["compound_id"]: stored[sid]["fingerprint"]
                    for sid in ids
                }
                compound_ids = list(fingerprints)
                matrix = pairwise_similarity(
                    [fingerprints[cid] for cid in compound_ids],
                    self.config.chemical_metric,
                )
                result.compound_similarity = pd.DataFrame(
                    matrix,
                    index=compound_ids,
                    columns=compound_ids,
                )
                result.compound_similarity.index.name = "compound_id"

            self._apply_aggregation(
                result,
                rows,
                stored,
            )

            result.status = "ok" if rows else "no_hits"

        return results

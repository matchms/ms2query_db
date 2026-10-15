"""Public Python API for building, opening, and querying MS2Query-DB libraries.

The rest of the package contains deliberately small implementation layers for
PostgreSQL storage, native matchms indexes, chemical search, and query
orchestration.  This module is the stable entry point that application code is
expected to use.

The two most common operations are::

    report = create_library(...)

and::

    with MS2QueryLibrary.open(...) as library:
        result = library.query_file(...)

Database connections owned by this API are always closed when an operation
finishes or when opening a library fails.
"""
from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from .builder import LibraryBuilder
from .chemistry import ChemapFingerprinter, FingerprintProjector, identify_compound
from .config import BuildConfig, QueryConfig, methods_tuple
from .db import Database
from .embeddings import MS2DeepScoreEmbedder
from .errors import CompatibilityError, ContractError, RejectedRecord
from .io import (
    LoadedSpectrumBatch,
    discover_sources,
    iter_source_batches,
    load_mgf_batch,
)
from .processing import collection_from_records
from .query import CandidateReranker, QueryEngine
from .records import InputRecord
from .results import BatchQueryResult
from .search import ChemicalSearch, NativeIndexStore
from .util import digest_json, scientific_versions

if TYPE_CHECKING:
    from matchms import SpectraCollection, Spectrum


DEFAULT_INDEX_DIR = "ms2query_indexes"
DEFAULT_SCHEMA = BuildConfig().schema
_SCIENCE_PACKAGES = ("matchms", "chemap", "rdkit", "ms2deepscore")


def _resolve_embedder(
    model_path: str | Path | None,
    embedder,
    *,
    allow_legacy: bool,
):
    """Return one embedder while rejecting ambiguous model configuration."""
    if model_path is not None and embedder is not None:
        raise ValueError("Pass model_path or embedder, not both.")
    if model_path is None:
        return embedder
    return MS2DeepScoreEmbedder.from_file(
        model_path,
        allow_legacy=allow_legacy,
    )


def _validate_classical_backends(config: BuildConfig) -> None:
    """Fail early when the installed matchms lacks a requested index backend."""
    from .search.native import make_scorer, require_index_api

    for method in config.classical_methods:
        scorer = make_scorer(method, config.scoring)
        require_index_api(scorer)


def _records_from_loaded_batch(
    loaded: LoadedSpectrumBatch,
) -> list[InputRecord]:
    """Convert a parsed MGF batch into query records without losing source IDs.

    ``load_mgf_batch`` keeps successful spectra inside a ``SpectraCollection``
    and keeps parse failures separately. ``QueryEngine`` currently consumes
    ``InputRecord`` objects, so this helper joins the two views back together in
    physical MGF-record order.  Only this API boundary materializes individual
    ``Spectrum`` rows; parsing and scientific preprocessing remain collection
    based.
    """
    by_index: dict[int, InputRecord] = {
        index: InputRecord(index, error=error)
        for index, error in loaded.rejected.items()
    }

    if loaded.collection is not None:
        for row, source_index in enumerate(loaded.source_indices):
            if source_index in by_index:
                raise ContractError(
                    f"Parsed query source index {source_index} is duplicated."
                )
            by_index[source_index] = InputRecord(
                source_index,
                loaded.collection[row],
            )

    return [by_index[index] for index in sorted(by_index)]


def create_library(
    source: str | Path,
    database_url: str,
    model_path: str | Path | None = None,
    *,
    chemical_csv: str | Path | None = None,
    config: BuildConfig | None = None,
    index_dir: str | Path = DEFAULT_INDEX_DIR,
    resume: bool = False,
    allow_legacy: bool = False,
    embedder=None,
    fingerprinter=None,
):
    """Build one immutable MS2Query-DB reference library.

    Parameters
    ----------
    source
        One MGF file or a directory containing reference MGF files.
    database_url
        PostgreSQL connection URL.
    model_path
        Optional MS2DeepScore ``.pt`` or ``.onnx`` model. Omit this for a
        classical-only library. ``embedder`` can be supplied instead for custom
        integrations and tests.
    chemical_csv
        Optional compound metadata table imported before the reference spectra.
    config
        Build settings. When omitted, :class:`BuildConfig` defaults are used.
    index_dir
        Root directory for local sharded matchms indexes.
    resume
        Resume an interrupted build. Resume requires the same configuration,
        input files, scientific package versions, and embedding model.
    allow_legacy
        Explicit opt-in for legacy PyTorch model loading where supported by
        MS2DeepScore. Only use this for a trusted model file.
    embedder, fingerprinter
        Optional dependency injection points for advanced use and testing.

    Returns
    -------
    BuildReport
        Library identifier and import statistics.

    Notes
    -----
    PostgreSQL ingestion is completed before local native indexes are built.
    Existing committed import batches remain resumable if a later index build
    fails.
    """
    if config is not None and not isinstance(config, BuildConfig):
        raise TypeError("config must be a BuildConfig instance or None.")

    cfg = config or BuildConfig()
    embedder = _resolve_embedder(
        model_path,
        embedder,
        allow_legacy=allow_legacy,
    )

    # Catch a missing/incompatible matchms indexed-search API before a long
    # reference import starts.
    _validate_classical_backends(cfg)

    database = Database(database_url, cfg.schema)
    try:
        builder = LibraryBuilder(
            database,
            cfg,
            embedder=embedder,
            fingerprinter=fingerprinter,
        )
        report = builder.build(
            source,
            chemical_csv,
            resume=resume,
        )

        if cfg.classical_methods:
            NativeIndexStore(database, index_dir).build(
                cfg.classical_methods
            )

        return report
    finally:
        database.close()


class MS2QueryLibrary:
    """An opened reference library ready for querying and data access.

    Instances own their :class:`Database` connection pool and should normally be
    used as context managers::

        with MS2QueryLibrary.open(dsn, model_path=...) as library:
            result = library.query_file("queries.mgf")

    Opening a library validates the stored fingerprint projection, embedding
    model identity (when supplied), and scientific package versions. Local
    native indexes are opened lazily by :class:`NativeIndexStore`.
    """

    def __init__(
        self,
        database: Database,
        info: dict[str, Any],
        *,
        embedder=None,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        fingerprinter=None,
        reranker: CandidateReranker | None = None,
    ) -> None:
        self.database = database
        self.info = info
        self.embedder = embedder
        self.reranker = reranker
        self.config = database.config
        self._closed = False

        self._validate_ready_state()
        self._validate_embedding_model()
        self._validate_scientific_versions()

        projection = self._validated_projection_bytes()
        self.projector = FingerprintProjector.from_bytes(
            projection,
            self.config.fingerprints,
        )
        self.fingerprinter = fingerprinter or ChemapFingerprinter(
            self.config.fingerprints
        )
        self.native = NativeIndexStore(database, index_dir)

    @classmethod
    def open(
        cls,
        database_url: str,
        model_path: str | Path | None = None,
        *,
        schema: str | None = None,
        index_dir: str | Path = DEFAULT_INDEX_DIR,
        allow_legacy: bool = False,
        embedder=None,
        fingerprinter=None,
        reranker: CandidateReranker | None = None,
    ) -> Self:
        """Open an existing library and validate its runtime compatibility.

        ``model_path`` is only required when a query configuration uses
        MS2DeepScore. A classical-only workflow can open a library without
        installing/loading MS2DeepScore even when embeddings are stored.
        """
        embedder = _resolve_embedder(
            model_path,
            embedder,
            allow_legacy=allow_legacy,
        )
        database = Database(database_url, schema or DEFAULT_SCHEMA)
        try:
            info = database.open()
            return cls(
                database,
                info,
                embedder=embedder,
                index_dir=index_dir,
                fingerprinter=fingerprinter,
                reranker=reranker,
            )
        except BaseException:
            database.close()
            raise

    def _validate_ready_state(self) -> None:
        if self.info.get("state") != "ready":
            raise CompatibilityError(
                "Library is not ready for queries. Resume or complete its "
                "build first."
            )

    def _validate_embedding_model(self) -> None:
        expected = self.info["descriptor"].get("model")
        if self.embedder is None:
            return
        if expected is None:
            raise CompatibilityError(
                "An embedding model was supplied, but this library was built "
                "without stored spectrum embeddings."
            )
        if (
            expected.get("model_id") != self.embedder.model_id
            or int(expected.get("dimension", -1)) != self.embedder.dimension
        ):
            raise CompatibilityError(
                "The supplied MS2DeepScore model does not match the model used "
                "to build this library."
            )

    def _validate_scientific_versions(self) -> None:
        """Require the scientific stack recorded by the library build.

        Persisted fingerprints, projections, embeddings, and native similarity
        indexes are scientific artifacts. Requiring the recorded package
        versions keeps results reproducible instead of silently mixing artifacts
        generated by different implementations.
        """
        expected_packages = self.info["descriptor"].get("packages", {})
        current_packages = scientific_versions()

        for name in _SCIENCE_PACKAGES:
            expected = expected_packages.get(name)
            if expected is None:
                continue
            # Classical-only querying does not need MS2DeepScore to be installed.
            if name == "ms2deepscore" and self.embedder is None:
                continue
            current = current_packages.get(name)
            if current != expected:
                raise CompatibilityError(
                    f"{name} version differs from the library build "
                    f"({expected!r} != {current!r})."
                )

    def _validated_projection_bytes(self) -> bytes:
        payload = bytes(self.info["projection"])
        actual = hashlib.sha256(payload).hexdigest()
        expected = self.info["projection_sha256"]
        if actual != expected:
            raise CompatibilityError(
                "Stored fingerprint-projection checksum does not match."
            )
        return payload

    def close(self) -> None:
        """Close the PostgreSQL connection pool owned by this library."""
        if not self._closed:
            self.database.close()
            self._closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc_info) -> None:
        self.close()

    def build_indexes(
        self,
        methods: str | Iterable[str] = "all",
        *,
        rebuild: bool = False,
    ) -> dict:
        """Build selected local classical indexes for this library.

        ``ms2deepscore`` is ignored because embedding retrieval uses PostgreSQL
        pgvector rather than a local native index. With ``rebuild=True`` the
        selected local artifacts are archived first; PostgreSQL data are never
        deleted.
        """
        selected = tuple(
            method
            for method in methods_tuple(methods)
            if method != "ms2deepscore"
        )
        if rebuild and selected:
            self.native.archive(selected)
        return self.native.build(selected)

    def _engine(
        self,
        config: QueryConfig,
        processor,
        reranker: CandidateReranker | None,
    ) -> QueryEngine:
        return QueryEngine(
            self.database,
            self.embedder,
            self.fingerprinter,
            self.native,
            self.projector,
            config,
            processor=processor,
            reranker=reranker if reranker is not None else self.reranker,
        )

    def _provenance(
        self,
        config: QueryConfig,
        processor=None,
    ) -> dict[str, Any]:
        """Describe the immutable library and settings behind query results."""
        return {
            "library_id": self.info["library_id"],
            "schema": self.database.schema,
            "library_descriptor_sha256": digest_json(
                self.info["descriptor"]
            ),
            "fingerprints": self.config.to_dict()["fingerprints"],
            "query_config": config.to_dict(),
            "processor": (
                "library_default"
                if processor is None
                else type(processor).__qualname__
            ),
            "ranking": (
                "retrieval ranks are method-specific; candidate aggregation "
                "does not create a fused spectral retrieval rank"
            ),
        }

    @staticmethod
    def _query_config(config: QueryConfig | None) -> QueryConfig:
        if config is None:
            return QueryConfig()
        if not isinstance(config, QueryConfig):
            raise TypeError("config must be a QueryConfig instance or None.")
        return config

    def query(
        self,
        spectra: Iterable["Spectrum"],
        *,
        config: QueryConfig | None = None,
        processor=None,
        reranker: CandidateReranker | None = None,
    ) -> BatchQueryResult:
        """Query an in-memory iterable of matchms spectra.

        Spectra are processed in bounded batches according to
        ``QueryConfig.batch_size``. Result ``query_index`` values correspond to
        positions in the supplied iterable.
        """
        cfg = self._query_config(config)
        engine = self._engine(cfg, processor, reranker)
        results = []
        batch: list[InputRecord] = []

        for index, spectrum in enumerate(spectra):
            batch.append(InputRecord(index, spectrum))
            if len(batch) == cfg.batch_size:
                results.extend(engine.process_batch(batch))
                batch = []

        if batch:
            results.extend(engine.process_batch(batch))

        return BatchQueryResult(
            results,
            self._provenance(cfg, processor),
        )

    def iter_query_file(
        self,
        path: str | Path,
        *,
        config: QueryConfig | None = None,
        processor=None,
        reranker: CandidateReranker | None = None,
    ):
        """Yield query results batch-by-batch from one MGF file.

        MGF record boundaries are isolated before matchms parsing. If one record
        cannot be parsed, that physical query index is rejected while later
        query IDs remain unchanged.
        """
        cfg = self._query_config(config)
        query_path = Path(path).expanduser().resolve()
        if not query_path.is_file():
            raise ValueError(
                "iter_query_file expects one existing MGF file; iterate over "
                "multiple files explicitly."
            )

        sources = discover_sources(query_path)
        if len(sources) != 1:
            raise ContractError("One query file must resolve to one source.")
        source = sources[0]
        engine = self._engine(cfg, processor, reranker)
        provenance = {
            **self._provenance(cfg, processor),
            "source": source.descriptor(),
        }

        for raw_records in iter_source_batches(
            source,
            cfg.batch_size,
        ):
            loaded = load_mgf_batch(
                raw_records,
                mz_precision=self.config.mz_precision,
            )
            records = _records_from_loaded_batch(loaded)
            results = engine.process_batch(
                records,
                source_label=source.path.name,
            )
            yield BatchQueryResult(results, provenance)

    def query_file(
        self,
        path: str | Path,
        *,
        config: QueryConfig | None = None,
        processor=None,
        reranker: CandidateReranker | None = None,
    ) -> BatchQueryResult:
        """Query one MGF file and combine all bounded batches in memory."""
        cfg = self._query_config(config)
        results = []
        provenance = self._provenance(cfg, processor)

        for batch in self.iter_query_file(
            path,
            config=cfg,
            processor=processor,
            reranker=reranker,
        ):
            results.extend(batch.results)
            provenance = batch.provenance

        return BatchQueryResult(results, provenance)

    def get_spectra(
        self,
        spectrum_ids: Sequence[int],
    ) -> SpectraCollection | None:
        """Load stored reference spectra as one ``SpectraCollection``.

        Input order and repeated IDs are preserved. An empty input returns
        ``None`` because there is no meaningful empty stored collection to
        reconstruct with the current matchms API.
        """
        ids = list(spectrum_ids)
        if not ids:
            return None
        records = self.database.get_candidates(ids)
        return collection_from_records(
            [records[spectrum_id] for spectrum_id in ids],
            mz_precision=self.config.mz_precision,
        )

    def get_compounds(
        self,
        compound_ids: Sequence[int],
    ) -> list[dict]:
        """Load decoded compound records in requested order, including repeats."""
        ids = list(compound_ids)
        if not ids:
            return []
        records = self.database.get_compounds(ids)
        return [records[compound_id] for compound_id in ids]

    def search_compounds(
        self,
        smiles: str,
        *,
        top_n: int = 20,
        pool: int = 500,
        metric: str = "minmax",
        exact: bool = False,
        exclude_self: bool = False,
    ):
        """Search the compound library by structure fingerprint similarity.

        Approximate mode uses the stored projection only to shortlist compounds;
        returned neighbors are always reranked with the full stored fingerprint
        and the requested exact chemical similarity metric.
        """
        identity = identify_compound(smiles)
        fingerprint = self.fingerprinter.compute_many(
            [identity.smiles]
        )[0]
        if isinstance(fingerprint, RejectedRecord):
            raise fingerprint

        return ChemicalSearch(
            self.database,
            self.projector,
        ).search(
            fingerprint,
            top_n=top_n,
            pool=pool,
            metric=metric,
            exact=exact,
            exclude_inchikey14=(
                identity.inchikey14 if exclude_self else None
            ),
        )


def library_statistics(
    database_url: str,
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    """Return library identity, state, and stored row/rejection counts."""
    database = Database(database_url, schema or DEFAULT_SCHEMA)
    try:
        info = database.open()
        return {
            "library_id": info["library_id"],
            "state": info["state"],
            **database.statistics(),
        }
    finally:
        database.close()


def doctor(
    database_url: str,
    *,
    schema: str | None = None,
) -> dict[str, Any]:
    """Collect a small diagnostic report for PostgreSQL and search backends.

    This does not modify the database. It is intended for installation checks
    and bug reports rather than scientific result generation.
    """
    from sqlalchemy import text

    from .search.native import make_scorer

    database = Database(database_url, schema or DEFAULT_SCHEMA)
    try:
        with database.engine.connect() as connection:
            status = dict(
                connection.execute(
                    text(
                        "SELECT current_database() AS database, "
                        "current_user AS username, "
                        "pg_is_in_recovery() AS recovery"
                    )
                ).mappings().one()
            )
            status["pgvector"] = connection.scalar(
                text(
                    "SELECT extversion FROM pg_extension "
                    "WHERE extname='vector'"
                )
            )

        status["packages"] = scientific_versions()
        try:
            status["matchms_indexed_methods"] = {}
            for method in (
                "cosine",
                "modified_cosine",
                "entropy_fragment",
                "entropy_hybrid",
            ):
                scorer = make_scorer(method, BuildConfig().scoring)
                status["matchms_indexed_methods"][method] = all(
                    callable(getattr(scorer, name, None))
                    for name in (
                        "build_index",
                        "search",
                        "save_index",
                        "load_index",
                    )
                )
        except ImportError as exc:
            status["matchms_error"] = str(exc)
        return status
    finally:
        database.close()

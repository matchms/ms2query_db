"""PostgreSQL lifecycle, atomic imports, and bounded library read services.

Each MS2Query-DB library owns one PostgreSQL schema. The database service keeps
transactions explicit, never silently migrates an existing schema, and exposes
bounded Core queries for builders, search indexes, and query-time candidate
materialization.

For combined positive/negative libraries, reference spectra always store their
actual ion mode. MS2DeepScore ANN indexes are built as separate partial HNSW
indexes per ion mode so same-mode queries never retrieve from the wrong mode and
filter afterwards.
"""
from __future__ import annotations

import hashlib
import logging
import re
import uuid
import zlib
from contextlib import contextmanager

import numpy as np
from sqlalchemy import (
    Float,
    bindparam,
    create_engine,
    func,
    inspect,
    select,
    text,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.schema import CreateSchema

from ..chemistry import decode_fingerprint
from ..config import BuildConfig, schema_name
from ..errors import CompatibilityError, ContractError
from ..records import RecordDecision
from ..util import chunks
from .models import models_for
from .repositories import (
    ImportRepository,
    ReferenceCompoundRepository,
    ReferenceSpectrumRepository,
)

logger = logging.getLogger(__name__)

# This is the first public ms2query_db schema format. It intentionally does not
# claim compatibility with the experimental ms2query-pg v2 schemas.
FORMAT_VERSION = 1
_VALID_SPECTRUM_IONMODES = frozenset({"positive", "negative"})


class Database:
    """Database service for one immutable MS2Query-DB library schema."""

    def __init__(self, dsn: str, schema: str):
        self.schema = schema_name(schema)

        url = make_url(dsn)
        if url.get_backend_name() != "postgresql":
            raise ValueError("A PostgreSQL DSN is required")
        url = url.set(drivername="postgresql+psycopg")

        self.engine = create_engine(
            url,
            pool_pre_ping=True,
            pool_size=2,
            max_overflow=1,
            connect_args={
                "connect_timeout": 15,
                "application_name": "ms2query-db",
            },
            execution_options={
                "schema_translate_map": {"library": self.schema}
            },
        )
        self.models = None
        self.config: BuildConfig | None = None

    def close(self) -> None:
        """Dispose all pooled PostgreSQL connections owned by this service."""
        self.engine.dispose()

    @contextmanager
    def build_lock(self):
        """Hold a schema-specific PostgreSQL advisory lock across build commits."""
        lock = int.from_bytes(
            hashlib.sha256(("ms2query:" + self.schema).encode()).digest()[:8],
            "big",
            signed=True,
        )

        with self.engine.connect() as conn:
            acquired = conn.scalar(
                text("SELECT pg_try_advisory_lock(:key)"),
                {"key": lock},
            )
            conn.commit()
            if not acquired:
                raise RuntimeError(
                    "Another builder/index writer owns this schema"
                )

            try:
                yield
            finally:
                if not conn.invalidated:
                    conn.execute(
                        text("SELECT pg_advisory_unlock(:key)"),
                        {"key": lock},
                    )
                    conn.commit()

    def initialize(
        self,
        config: BuildConfig,
        descriptor: dict,
        projector,
        *,
        resume: bool,
    ) -> dict:
        """Create a fresh library schema or strictly resume an identical build.

        Existing schemas are never dropped, overwritten, or silently migrated.
        Resume requires the identical immutable build descriptor.
        """
        dimension = (
            descriptor["model"]["dimension"]
            if descriptor["model"]
            else 1
        )
        self.models = models_for(
            dimension,
            config.fingerprints.projection_dim,
        )
        self.config = config

        with self.engine.begin() as conn:
            exists = inspect(conn).has_schema(self.schema)
            if exists:
                if not resume:
                    raise FileExistsError(
                        f"Schema {self.schema!r} already exists; use --resume "
                        "or a new schema"
                    )
                if not inspect(conn).has_table(
                    "library_info",
                    schema=self.schema,
                ):
                    raise CompatibilityError(
                        "Existing schema is not an MS2Query-DB library; keep "
                        "it and choose a new schema"
                    )

                row = conn.execute(
                    select(self.models.LibraryInfo.__table__)
                ).mappings().one()
                if row["format_version"] != FORMAT_VERSION:
                    raise CompatibilityError(
                        "Existing schema uses a different MS2Query-DB format"
                    )
                if row["descriptor"] != descriptor:
                    raise CompatibilityError(
                        "Resume requires identical configuration, inputs, "
                        "scientific versions, and embedding model"
                    )
                return dict(row)

            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            extension = conn.scalar(
                text(
                    "SELECT extversion FROM pg_extension "
                    "WHERE extname = 'vector'"
                )
            )
            if extension is None:
                raise CompatibilityError(
                    "Could not install or locate the pgvector extension"
                )

            version = tuple(
                int(part)
                for part in re.findall(r"\d+", extension)[:3]
            )
            if version < (0, 8, 0):
                raise CompatibilityError(
                    "pgvector >= 0.8.0 is required; upgrade before building"
                )

            conn.execute(CreateSchema(self.schema))
            self.models.Base.metadata.create_all(conn)

            projection = projector.to_bytes()
            values = {
                "id": 1,
                "library_id": str(uuid.uuid4()),
                "format_version": FORMAT_VERSION,
                "state": "importing",
                "descriptor": descriptor,
                "projection": projection,
                "projection_sha256": hashlib.sha256(
                    projection
                ).hexdigest(),
                "last_error": None,
            }
            conn.execute(
                self.models.LibraryInfo.__table__.insert().values(**values)
            )
            return values

    def open(self, *, require_ready: bool = True) -> dict:
        """Open an existing library and instantiate its dimensioned mappings."""
        quoted = self.engine.dialect.identifier_preparer.quote_schema(
            self.schema
        )
        with self.engine.connect() as conn:
            if not inspect(conn).has_table(
                "library_info",
                schema=self.schema,
            ):
                raise CompatibilityError(
                    "Library not found or schema is not an MS2Query-DB library"
                )
            info = dict(
                conn.execute(
                    text(
                        f"SELECT * FROM {quoted}.library_info WHERE id = 1"
                    )
                ).mappings().one()
            )

        if info["format_version"] != FORMAT_VERSION:
            raise CompatibilityError(
                "Unsupported MS2Query-DB library format"
            )

        self.config = BuildConfig.from_dict(
            info["descriptor"]["config"]
        )
        model = info["descriptor"]["model"]
        self.models = models_for(
            model["dimension"] if model else 1,
            self.config.fingerprints.projection_dim,
        )

        if require_ready and info["state"] != "ready":
            raise CompatibilityError(
                f"Library {self.schema!r} is not ready "
                f"(state={info['state']!r})"
            )

        return info

    def set_state(self, state: str, error: str | None = None) -> None:
        """Update the singleton library build state."""
        with self.engine.begin() as conn:
            conn.execute(
                update(self.models.LibraryInfo)
                .where(self.models.LibraryInfo.id == 1)
                .values(state=state, last_error=error)
            )

    def source_id(self, source) -> int:
        """Resolve a stable database ID for an immutable input source."""
        from sqlalchemy.dialects.postgresql import insert

        table = self.models.ImportSource.__table__
        with self.engine.begin() as conn:
            conn.execute(
                insert(table)
                .values(
                    name=source.path.name,
                    sha256=source.sha256,
                    kind=source.kind,
                )
                .on_conflict_do_nothing(
                    index_elements=[table.c.kind, table.c.sha256]
                )
            )
            source_id = conn.scalar(
                select(table.c.id).where(
                    table.c.sha256 == source.sha256,
                    table.c.kind == source.kind,
                )
            )

        if source_id is None:
            raise ContractError("Could not resolve import source ID")
        return int(source_id)

    def completed_batches(self, source_id: int) -> dict[int, int]:
        """Return committed batch starts and consumed row counts for resume."""
        table = self.models.ImportBatch.__table__
        with self.engine.connect() as conn:
            return {
                int(row.first_index): int(row.consumed)
                for row in conn.execute(
                    select(
                        table.c.first_index,
                        table.c.consumed,
                    ).where(table.c.source_id == source_id)
                )
            }

    def commit_batch(
        self,
        source_id: int,
        ledger,
        prepared,
        *,
        reference: bool,
    ) -> None:
        """Atomically commit compounds, spectra, decisions, and checkpoint.

        ``prepared`` contains only fully validated records. Duplicate molecular
        structures resolve to one compound but each accepted source record still
        receives its own import decision and, for reference input, spectrum row.
        """
        with self.engine.begin() as conn:
            compounds = ReferenceCompoundRepository(
                conn,
                self.models,
            ).upsert_many([row["compound"] for row in prepared])

            spectra = []
            for row in prepared:
                key = row["compound"]["identity_key"]
                if key not in compounds:
                    raise ContractError(
                        "Compound repository lost an accepted identity"
                    )
                if reference:
                    spectra.append(
                        dict(
                            row["spectrum"],
                            source_id=source_id,
                            source_index=row["index"],
                            compound_id=compounds[key],
                        )
                    )

            ids = (
                ReferenceSpectrumRepository(
                    conn,
                    self.models,
                ).insert_many(spectra)
                if reference
                else {}
            )

            for row in prepared:
                key = row["compound"]["identity_key"]
                ledger.add(
                    RecordDecision(
                        row["index"],
                        True,
                        compound_id=compounds[key],
                        spectrum_id=(
                            ids[row["index"]]
                            if reference
                            else None
                        ),
                        detail=row.get("detail", {}),
                    )
                )

            ImportRepository(
                conn,
                self.models,
            ).finish(source_id, ledger)

    def _spectra_by_ionmode(self) -> dict[str, int]:
        """Return accepted reference-spectrum counts by actual ion mode."""
        table = self.models.ReferenceSpectrum.__table__
        with self.engine.connect() as conn:
            return {
                str(mode): int(count)
                for mode, count in conn.execute(
                    select(
                        table.c.ionmode,
                        func.count(),
                    ).group_by(table.c.ionmode)
                )
            }

    def _embedding_index_modes(self) -> tuple[str, ...]:
        """Return ion modes for which an embedding HNSW index is required."""
        counts = self._spectra_by_ionmode()
        unexpected = set(counts) - _VALID_SPECTRUM_IONMODES
        if unexpected:
            raise ContractError(
                "Reference spectra contain invalid ion modes: "
                f"{sorted(unexpected)}"
            )

        if self.config.ionmode == "both":
            missing = _VALID_SPECTRUM_IONMODES - set(counts)
            if missing:
                raise CompatibilityError(
                    "BuildConfig(ionmode='both') requires accepted reference "
                    "spectra in both modes; missing "
                    f"{sorted(missing)}"
                )
            return ("positive", "negative")

        if set(counts) - {self.config.ionmode}:
            raise ContractError(
                "Single-mode library contains reference spectra from another "
                "ion mode"
            )

        return (
            (self.config.ionmode,)
            if counts.get(self.config.ionmode, 0)
            else ()
        )

    def finalize(self) -> None:
        """Build PostgreSQL ANN indexes after ingestion and mark library ready.

        Chemical fingerprints use one global HNSW index. Spectrum embeddings
        use one partial HNSW index per actual ion mode. The sum of the partial
        index sizes is roughly comparable to one global index while preserving
        correct and efficient same-mode top-N semantics.
        """
        preparer = self.engine.dialect.identifier_preparer
        schema = preparer.quote_schema(self.schema)
        cfg = self.config

        modes = self._embedding_index_modes()
        table = self.models.ReferenceSpectrum.__table__
        with self.engine.connect() as conn:
            has_embeddings = (
                conn.scalar(
                    select(func.count())
                    .select_from(table)
                    .where(table.c.embedding.is_not(None))
                )
                > 0
            )

        indexes: list[tuple[str, str, str, str | None]] = [
            (
                "chemical_projection_hnsw",
                "reference_compounds",
                "fingerprint_projection",
                None,
            )
        ]

        if has_embeddings:
            for mode in modes:
                indexes.append(
                    (
                        f"spectrum_embedding_{mode}_hnsw",
                        "reference_spectra",
                        "embedding",
                        f"ionmode = '{mode}' AND embedding IS NOT NULL",
                    )
                )

        self.set_state("indexing")

        for name, table_name, column, predicate in indexes:
            logger.info(
                "Building PostgreSQL index %s (completed indexes are reused)",
                name,
            )
            where = f" WHERE {predicate}" if predicate else ""
            with self.engine.begin() as conn:
                conn.execute(
                    text(
                        "SELECT set_config('maintenance_work_mem', :value, true)"
                    ),
                    {"value": f"{cfg.maintenance_work_mem_mb}MB"},
                )
                # Avoid multiplying index-build memory by parallel workers.
                conn.execute(
                    text(
                        "SELECT set_config(" 
                        "'max_parallel_maintenance_workers', '0', true)"
                    )
                )
                conn.execute(
                    text(
                        f"CREATE INDEX IF NOT EXISTS {preparer.quote(name)} "
                        f"ON {schema}.{preparer.quote(table_name)} USING hnsw "
                        f"({preparer.quote(column)} vector_cosine_ops) "
                        f"WITH (m={cfg.hnsw_m}, "
                        f"ef_construction={cfg.hnsw_ef_construction})"
                        f"{where}"
                    )
                )

        with self.engine.begin() as conn:
            conn.execute(text(f"ANALYZE {schema}.reference_spectra"))
            conn.execute(text(f"ANALYZE {schema}.reference_compounds"))

        self.set_state("ready")

    def info(self) -> dict:
        """Return the singleton library descriptor row."""
        with self.engine.connect() as conn:
            return dict(
                conn.execute(
                    select(self.models.LibraryInfo.__table__)
                ).mappings().one()
            )

    def statistics(self) -> dict:
        """Return compact library/import statistics, including ion-mode counts."""
        result = {}
        with self.engine.connect() as conn:
            for name, model in (
                ("compounds", self.models.ReferenceCompound),
                ("spectra", self.models.ReferenceSpectrum),
                ("batches", self.models.ImportBatch),
            ):
                result[name] = int(
                    conn.scalar(
                        select(func.count()).select_from(model)
                    )
                )

            decisions = self.models.ImportRecord.__table__
            result["records_by_status"] = {
                str(status): int(count)
                for status, count in conn.execute(
                    select(
                        decisions.c.status,
                        func.count(),
                    ).group_by(decisions.c.status)
                ).all()
            }
            result["rejections_by_reason"] = {
                str(reason): int(count)
                for reason, count in conn.execute(
                    select(
                        decisions.c.reason_code,
                        func.count(),
                    )
                    .where(decisions.c.status == "rejected")
                    .group_by(decisions.c.reason_code)
                ).all()
            }

            spectra = self.models.ReferenceSpectrum.__table__
            result["spectra_by_ionmode"] = {
                str(mode): int(count)
                for mode, count in conn.execute(
                    select(
                        spectra.c.ionmode,
                        func.count(),
                    ).group_by(spectra.c.ionmode)
                ).all()
            }

        return result

    def _normalize_ionmode(self, ionmode: str | None) -> str | None:
        """Validate an optional spectrum ion-mode restriction."""
        if ionmode is None:
            if self.config.ionmode == "both":
                return None
            return self.config.ionmode

        if ionmode not in _VALID_SPECTRUM_IONMODES:
            raise ValueError("ionmode must be 'positive' or 'negative'")

        if (
            self.config.ionmode != "both"
            and ionmode != self.config.ionmode
        ):
            raise ValueError(
                f"Library contains {self.config.ionmode!r} spectra, not "
                f"{ionmode!r} spectra"
            )

        return ionmode

    def iter_spectra(
        self,
        batch_size: int = 1000,
        *,
        ionmode: str | None = None,
    ):
        """Yield keyset-paged reference rows without hydrating embeddings.

        Parameters
        ----------
        batch_size
            Maximum rows yielded per database round trip.
        ionmode
            Optional actual spectrum mode. For a single-mode library, omitted
            mode defaults to the library mode. For a combined library, omitted
            mode iterates both modes; native index construction should pass an
            explicit mode and build separate shards.
        """
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        mode = self._normalize_ionmode(ionmode)
        table = self.models.ReferenceSpectrum.__table__
        columns = [
            column
            for column in table.c
            if column.name != "embedding"
        ]
        last = 0

        while True:
            stmt = (
                select(*columns)
                .where(table.c.id > last)
                .order_by(table.c.id)
                .limit(batch_size)
            )
            if mode is not None:
                stmt = stmt.where(table.c.ionmode == mode)

            with self.engine.connect() as conn:
                rows = [
                    dict(row)
                    for row in conn.execute(stmt).mappings()
                ]

            if not rows:
                break

            yield rows
            last = rows[-1]["id"]

    def get_candidates(self, spectrum_ids) -> dict[int, dict]:
        """Return decoded spectrum+compound rows keyed by spectrum ID."""
        ids = {int(value) for value in spectrum_ids}
        if not ids:
            return {}

        spectra = self.models.ReferenceSpectrum.__table__
        compounds = self.models.ReferenceCompound.__table__
        columns = [
            spectra.c.id.label("spectrum_id"),
            *[
                column
                for column in spectra.c
                if column.name
                not in {"id", "source_id", "source_index"}
            ],
            *[
                column
                for column in compounds.c
                if column.name
                not in {"id", "fingerprint_projection"}
            ],
        ]

        result = {}
        with self.engine.connect() as conn:
            for block in chunks(sorted(ids), 1000):
                stmt = (
                    select(*columns)
                    .select_from(
                        spectra.join(
                            compounds,
                            spectra.c.compound_id == compounds.c.id,
                        )
                    )
                    .where(spectra.c.id.in_(block))
                )
                for row in conn.execute(stmt).mappings():
                    item = dict(row)
                    item["fingerprint"] = self._decode_stored_fingerprint(
                        item["compound_id"],
                        item["fingerprint"],
                        spectrum_id=item["spectrum_id"],
                    )
                    result[int(item["spectrum_id"])] = item

        if set(result) != ids:
            raise ContractError(
                "Search index refers to missing spectrum IDs"
            )
        return result

    def get_compounds(self, compound_ids) -> dict[int, dict]:
        """Return decoded compound rows keyed by database ID."""
        ids = {int(value) for value in compound_ids}
        if not ids:
            return {}

        table = self.models.ReferenceCompound.__table__
        result = {}
        with self.engine.connect() as conn:
            for block in chunks(sorted(ids), 1000):
                stmt = select(
                    *[
                        column
                        for column in table.c
                        if column.name != "fingerprint_projection"
                    ]
                ).where(table.c.id.in_(block))
                for row in conn.execute(stmt).mappings():
                    item = self._decode_compound(row)
                    result[item["compound_id"]] = item

        if set(result) != ids:
            raise KeyError(
                "Unknown compound IDs: "
                f"{sorted(ids - set(result))[:10]}"
            )
        return result

    def search_embeddings(
        self,
        embeddings,
        top_n: int,
        *,
        ionmode: str | None = None,
        ef_search: int = 200,
        exact: bool = False,
    ) -> list[list[dict]]:
        """Search MS2DeepScore reference embeddings with optional mode routing.

        A combined library should pass ``ionmode='positive'`` or ``'negative'``
        for approximate search so PostgreSQL can use the corresponding partial
        HNSW index. QueryEngine already performs this split for same-mode search.
        Cross-mode approximate retrieval is intentionally not implemented yet.
        """
        if top_n < 1:
            raise ValueError("top_n must be positive")
        if ef_search < 1:
            raise ValueError("ef_search must be positive")

        mode = self._normalize_ionmode(ionmode)
        if (
            self.config.ionmode == "both"
            and mode is None
            and not exact
        ):
            raise NotImplementedError(
                "Approximate cross-mode embedding search is not implemented. "
                "Pass ionmode='positive'/'negative' or use exact=True."
            )

        table = self.models.ReferenceSpectrum.__table__
        result: list[list[dict]] = []

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "SELECT set_config('hnsw.ef_search', :n, true)"
                ),
                {"n": str(min(1000, max(ef_search, top_n)))},
            )
            conn.execute(
                text(
                    "SELECT set_config(" 
                    "'hnsw.iterative_scan', 'strict_order', true)"
                )
            )

            if exact:
                conn.execute(text("SET LOCAL enable_indexscan = off"))
                conn.execute(text("SET LOCAL enable_bitmapscan = off"))

            for embedding in embeddings:
                vector = np.asarray(embedding, dtype=np.float32)
                if vector.ndim != 1 or not vector.size:
                    raise ValueError(
                        "Embedding queries must be nonempty one-dimensional "
                        "vectors"
                    )
                if not np.isfinite(vector).all():
                    raise ValueError(
                        "Embedding queries must contain only finite values"
                    )

                distance = table.c.embedding.op(
                    "<=>",
                    return_type=Float,
                )(
                    bindparam(
                        "embedding",
                        type_=table.c.embedding.type,
                    )
                )

                stmt = (
                    select(
                        table.c.id,
                        distance.label("distance"),
                    )
                    .where(table.c.embedding.is_not(None))
                    .order_by(distance)
                    .limit(top_n)
                )
                if mode is not None:
                    stmt = stmt.where(table.c.ionmode == mode)

                hits = [
                    {
                        "spectrum_id": int(row.id),
                        "score": float(
                            np.clip(1 - row.distance, -1, 1)
                        ),
                        "matches": None,
                    }
                    for row in conn.execute(
                        stmt,
                        {"embedding": vector},
                    )
                ]
                hits.sort(
                    key=lambda row: (
                        -row["score"],
                        row["spectrum_id"],
                    )
                )
                result.append(hits)

        return result

    def chemical_candidates(
        self,
        projection,
        limit: int,
        *,
        ef_search: int = 500,
    ) -> list[dict]:
        """Return projected chemical ANN candidates with decoded fingerprints."""
        if limit < 1:
            raise ValueError("limit must be positive")

        table = self.models.ReferenceCompound.__table__
        distance = table.c.fingerprint_projection.op(
            "<=>",
            return_type=Float,
        )(
            bindparam(
                "projection",
                type_=table.c.fingerprint_projection.type,
            )
        )

        with self.engine.begin() as conn:
            conn.execute(
                text(
                    "SELECT set_config('hnsw.ef_search', :n, true)"
                ),
                {"n": str(min(1000, max(limit, ef_search)))},
            )
            conn.execute(
                text(
                    "SELECT set_config(" 
                    "'hnsw.iterative_scan', 'strict_order', true)"
                )
            )

            rows = conn.execute(
                select(
                    *[
                        column
                        for column in table.c
                        if column.name != "fingerprint_projection"
                    ]
                )
                .order_by(distance)
                .limit(limit),
                {"projection": projection},
            ).mappings().all()

        return [self._decode_compound(row) for row in rows]

    def iter_compounds(self, batch_size: int = 1000):
        """Yield decoded compounds using keyset pagination."""
        if batch_size < 1:
            raise ValueError("batch_size must be positive")

        table = self.models.ReferenceCompound.__table__
        last = 0
        while True:
            with self.engine.connect() as conn:
                rows = conn.execute(
                    select(
                        *[
                            column
                            for column in table.c
                            if column.name != "fingerprint_projection"
                        ]
                    )
                    .where(table.c.id > last)
                    .order_by(table.c.id)
                    .limit(batch_size)
                ).mappings().all()
                rows = [
                    self._decode_compound(row)
                    for row in rows
                ]

            if not rows:
                break

            yield rows
            last = rows[-1]["compound_id"]

    def _decode_compound(self, row) -> dict:
        """Convert a database compound row to the public decoded representation."""
        item = dict(row)
        item["compound_id"] = int(item.pop("id"))
        item["fingerprint"] = self._decode_stored_fingerprint(
            item["compound_id"],
            item["fingerprint"],
        )
        return item

    def _decode_stored_fingerprint(
        self,
        compound_id: int,
        payload,
        *,
        spectrum_id: int | None = None,
    ):
        """Decode a stored fingerprint, retrying once after a fresh DB read.

        The retry is intentionally narrow and diagnostic. Persistent corruption
        is converted to a clear ContractError; a successful fresh read is logged
        because an initial decode failure should not normally occur.
        """
        try:
            return decode_fingerprint(
                payload,
                self.config.fingerprints.size,
            )
        except (zlib.error, ContractError) as first_exc:
            context = f"compound_id={compound_id}"
            if spectrum_id is not None:
                context += f", spectrum_id={spectrum_id}"

            logger.warning(
                "Fingerprint decode failed for %s; retrying once using a fresh "
                "database connection: %s",
                context,
                first_exc,
            )

            table = self.models.ReferenceCompound.__table__
            with self.engine.connect() as retry_conn:
                retry_payload = retry_conn.scalar(
                    select(table.c.fingerprint).where(
                        table.c.id == compound_id
                    )
                )

            if retry_payload is None:
                raise ContractError(
                    "Compound disappeared during fingerprint retry: "
                    f"{context}"
                ) from first_exc

            try:
                fingerprint = decode_fingerprint(
                    retry_payload,
                    self.config.fingerprints.size,
                )
            except (zlib.error, ContractError) as retry_exc:
                raise ContractError(
                    "Stored fingerprint could not be decoded after a fresh "
                    f"database read: {context}. First error: {first_exc}; "
                    f"retry error: {retry_exc}"
                ) from retry_exc

            logger.warning(
                "Fresh fingerprint read succeeded for %s; the initial decode "
                "failure was transient.",
                context,
            )
            return fingerprint

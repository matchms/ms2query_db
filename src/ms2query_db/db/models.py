"""SQLAlchemy mappings for one MS2Query-DB library schema.

A small model factory binds vector dimensions per library without global mutable
state. Tables use the placeholder schema name ``"library"``; the Database
engine translates that placeholder to the configured PostgreSQL schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    ForeignKeyConstraint,
    Identity,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from .types import Vector


@dataclass(frozen=True)
class ModelSet:
    """Library-specific SQLAlchemy mappings with fixed vector dimensions."""

    Base: type
    LibraryInfo: type
    ReferenceCompound: type
    ReferenceSpectrum: type
    ImportSource: type
    ImportBatch: type
    ImportRecord: type


@lru_cache(maxsize=32)
def models_for(
    embedding_dim: int,
    projection_dim: int,
) -> ModelSet:
    """Create or reuse mappings for one embedding/projection dimension pair."""
    if not isinstance(embedding_dim, int) or not 1 <= embedding_dim <= 2000:
        raise ValueError("embedding_dim must be an integer in 1..2000")
    if not isinstance(projection_dim, int) or not 1 <= projection_dim <= 2000:
        raise ValueError("projection_dim must be an integer in 1..2000")

    class Base(DeclarativeBase):
        """Declarative base for this dimension-specific model set."""

    class LibraryInfo(Base):
        """Singleton descriptor and build state for one library schema."""

        __tablename__ = "library_info"
        __table_args__ = (
            CheckConstraint("id = 1", name="ck_library_info_singleton"),
            CheckConstraint(
                "state IN ('importing', 'indexing', 'ready', 'failed')",
                name="ck_library_info_state",
            ),
            {"schema": "library"},
        )

        id: Mapped[int] = mapped_column(Integer, primary_key=True)
        library_id: Mapped[str] = mapped_column(String(36), unique=True)
        format_version: Mapped[int] = mapped_column(Integer)
        state: Mapped[str] = mapped_column(String(32))
        descriptor: Mapped[dict] = mapped_column(JSONB)
        projection: Mapped[bytes] = mapped_column(LargeBinary)
        projection_sha256: Mapped[str] = mapped_column(String(64))
        last_error: Mapped[str | None] = mapped_column(Text)
        created_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
        )

    class ImportSource(Base):
        """Immutable source file identified by content hash and source kind."""

        __tablename__ = "import_sources"
        __table_args__ = (
            UniqueConstraint(
                "kind",
                "sha256",
                name="uq_import_source_kind_sha256",
            ),
            CheckConstraint(
                "kind IN ('spectra', 'chemicals')",
                name="ck_import_source_kind",
            ),
            {"schema": "library"},
        )

        id: Mapped[int] = mapped_column(
            BigInteger,
            Identity(),
            primary_key=True,
        )
        kind: Mapped[str] = mapped_column(String(16))
        sha256: Mapped[str] = mapped_column(String(64))
        name: Mapped[str] = mapped_column(Text)

    class ReferenceCompound(Base):
        """Unique molecular structure and its searchable chemical fingerprint."""

        __tablename__ = "reference_compounds"
        __table_args__ = ({"schema": "library"},)

        id: Mapped[int] = mapped_column(
            BigInteger,
            Identity(),
            primary_key=True,
        )
        identity_key: Mapped[str] = mapped_column(
            String(64),
            unique=True,
        )
        inchikey14: Mapped[str] = mapped_column(String(14), index=True)
        inchikey: Mapped[str] = mapped_column(String(27), index=True)
        smiles: Mapped[str] = mapped_column(Text)
        inchi: Mapped[str] = mapped_column(Text)
        formula: Mapped[str] = mapped_column(Text)
        exact_mass: Mapped[float] = mapped_column(Float(53))

        superclass: Mapped[str | None] = mapped_column(Text)
        compound_class: Mapped[str | None] = mapped_column(Text)
        subclass: Mapped[str | None] = mapped_column(Text)
        visual_x: Mapped[float | None] = mapped_column(Float(53))
        visual_y: Mapped[float | None] = mapped_column(Float(53))
        pubchem_cid: Mapped[int | None] = mapped_column(BigInteger)
        extra_metadata: Mapped[dict] = mapped_column(JSONB)

        # Large values are deferred for ORM callers. Core queries used by the
        # database service select them explicitly when required.
        fingerprint: Mapped[bytes] = mapped_column(
            LargeBinary,
            deferred=True,
        )
        fingerprint_projection: Mapped[list[float]] = mapped_column(
            Vector(projection_dim),
            deferred=True,
        )

        reference_spectra: Mapped[list["ReferenceSpectrum"]] = relationship(
            back_populates="reference_compound",
            lazy="raise",
        )

    class ReferenceSpectrum(Base):
        """Canonical reference MS/MS spectrum linked to one compound.

        ``ionmode`` always stores the actual spectrum mode (``positive`` or
        ``negative``), including in a library configured with ``ionmode='both'``.
        """

        __tablename__ = "reference_spectra"
        __table_args__ = (
            UniqueConstraint(
                "source_id",
                "source_index",
                name="uq_reference_spectrum_source_row",
            ),
            CheckConstraint(
                "n_peaks > 0",
                name="ck_reference_spectrum_has_peaks",
            ),
            CheckConstraint(
                "octet_length(mz_values) = 8 * n_peaks",
                name="ck_reference_spectrum_mz_payload",
            ),
            CheckConstraint(
                "octet_length(intensity_values) = 8 * n_peaks",
                name="ck_reference_spectrum_intensity_payload",
            ),
            CheckConstraint(
                "precursor_mz > 0",
                name="ck_reference_spectrum_precursor",
            ),
            CheckConstraint(
                "ionmode IN ('positive', 'negative')",
                name="ck_reference_spectrum_ionmode",
            ),
            {"schema": "library"},
        )

        id: Mapped[int] = mapped_column(
            BigInteger,
            Identity(),
            primary_key=True,
        )
        source_id: Mapped[int] = mapped_column(
            ForeignKey("library.import_sources.id")
        )
        source_index: Mapped[int] = mapped_column(BigInteger)
        compound_id: Mapped[int] = mapped_column(
            ForeignKey("library.reference_compounds.id"),
            index=True,
        )

        # Peak-only hashes are useful for diagnostics/deduplication but are not
        # a unique record identity: the same spectrum may legitimately occur in
        # several sources or for repeated measurements.
        spectrum_hash: Mapped[str] = mapped_column(Text, index=True)
        precursor_mz: Mapped[float] = mapped_column(Float(53))
        ionmode: Mapped[str] = mapped_column(String(8), index=True)
        retention_time: Mapped[float | None] = mapped_column(Float(53))
        n_peaks: Mapped[int] = mapped_column(Integer)
        mz_values: Mapped[bytes] = mapped_column(
            LargeBinary,
            deferred=True,
        )
        intensity_values: Mapped[bytes] = mapped_column(
            LargeBinary,
            deferred=True,
        )
        other_metadata: Mapped[dict] = mapped_column(JSONB)
        embedding: Mapped[list[float] | None] = mapped_column(
            Vector(embedding_dim),
            nullable=True,
            deferred=True,
        )

        reference_compound: Mapped["ReferenceCompound"] = relationship(
            back_populates="reference_spectra",
            lazy="raise",
        )

    class ImportBatch(Base):
        """One atomically committed source batch and its accounting totals."""

        __tablename__ = "import_batches"
        __table_args__ = (
            CheckConstraint(
                "consumed = accepted + rejected",
                name="ck_import_batch_accounting",
            ),
            CheckConstraint(
                "consumed >= 0 AND accepted >= 0 AND rejected >= 0",
                name="ck_import_batch_nonnegative",
            ),
            {"schema": "library"},
        )

        source_id: Mapped[int] = mapped_column(
            ForeignKey("library.import_sources.id"),
            primary_key=True,
        )
        first_index: Mapped[int] = mapped_column(
            BigInteger,
            primary_key=True,
        )
        consumed: Mapped[int] = mapped_column(Integer)
        accepted: Mapped[int] = mapped_column(Integer)
        rejected: Mapped[int] = mapped_column(Integer)
        committed_at: Mapped[datetime] = mapped_column(
            DateTime(timezone=True),
            server_default=func.now(),
        )

    class ImportRecord(Base):
        """Final accepted/rejected decision for one physical source record."""

        __tablename__ = "import_records"
        __table_args__ = (
            ForeignKeyConstraint(
                ["source_id", "batch_start"],
                [
                    "library.import_batches.source_id",
                    "library.import_batches.first_index",
                ],
            ),
            CheckConstraint(
                "status IN ('accepted', 'rejected')",
                name="ck_import_record_status",
            ),
            CheckConstraint(
                "(status = 'accepted' AND compound_id IS NOT NULL "
                "AND reason_code IS NULL) OR "
                "(status = 'rejected' AND reason_code IS NOT NULL "
                "AND compound_id IS NULL AND spectrum_id IS NULL)",
                name="ck_import_record_decision",
            ),
            {"schema": "library"},
        )

        source_id: Mapped[int] = mapped_column(
            BigInteger,
            primary_key=True,
        )
        source_index: Mapped[int] = mapped_column(
            BigInteger,
            primary_key=True,
        )
        batch_start: Mapped[int] = mapped_column(BigInteger)
        status: Mapped[str] = mapped_column(String(16))
        reason_code: Mapped[str | None] = mapped_column(Text, index=True)
        reason: Mapped[str | None] = mapped_column(Text)
        compound_id: Mapped[int | None] = mapped_column(
            ForeignKey("library.reference_compounds.id")
        )
        spectrum_id: Mapped[int | None] = mapped_column(
            ForeignKey("library.reference_spectra.id")
        )
        detail: Mapped[dict] = mapped_column(JSONB)

    return ModelSet(
        Base,
        LibraryInfo,
        ReferenceCompound,
        ReferenceSpectrum,
        ImportSource,
        ImportBatch,
        ImportRecord,
    )

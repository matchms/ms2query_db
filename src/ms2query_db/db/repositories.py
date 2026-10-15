"""Small database helpers used during library import.

The classes in this module are intentionally thin wrappers around SQLAlchemy
Core statements. They keep repetitive insert/upsert code out of
:mod:`ms2query_db.db.database`, while leaving transaction control in one place:
the caller.

A repository therefore **never commits or rolls back**.  It receives an open
SQLAlchemy connection and executes statements on that connection.  In the
normal build path, :class:`~ms2query_db.db.database.Database` owns the enclosing
batch transaction, so compounds, spectra, import decisions, and the batch
checkpoint either commit together or not at all.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert

from ..errors import ContractError
from ..util import chunks

# These columns describe optional annotations rather than molecular identity.
# When the same compound is seen again, an existing non-null value wins and a
# new value is used only to fill a missing database field.
FILLABLE_METADATA_COLUMNS = (
    "superclass",
    "compound_class",
    "subclass",
    "visual_x",
    "visual_y",
    "pubchem_cid",
)

COMPOUND_WRITE_BATCH_SIZE = 500
COMPOUND_ID_LOOKUP_BATCH_SIZE = 1000
SPECTRUM_WRITE_BATCH_SIZE = 500
IMPORT_RECORD_WRITE_BATCH_SIZE = 500


def _copy_compound_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Copy one compound row without sharing its mutable metadata dictionary."""
    copied = dict(row)
    copied["extra_metadata"] = dict(row.get("extra_metadata") or {})
    return copied


def _merge_missing_metadata(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
) -> None:
    """Fill metadata missing from *target* using another row of the same compound.

    ``identity_key`` is already known to be the same for both rows.  This helper
    deliberately does not overwrite an earlier non-null annotation: import
    order therefore has deterministic precedence when source files disagree.
    """
    for column in FILLABLE_METADATA_COLUMNS:
        if target.get(column) is None and incoming.get(column) is not None:
            target[column] = incoming[column]

    target_extra = target["extra_metadata"]
    for key, value in (incoming.get("extra_metadata") or {}).items():
        target_extra.setdefault(key, value)


class ReferenceCompoundRepository:
    """Insert compounds and resolve their database IDs.

    Several input records can refer to the same molecular structure.  The
    database stores that structure once, keyed by ``identity_key``.  This class
    performs two related jobs:

    1. combine duplicate compound rows from the current import batch;
    2. insert new compounds or enrich existing compounds with metadata that was
       previously missing.

    Existing metadata is never replaced by a later import.  The method returns
    a mapping from ``identity_key`` to the database ``compound_id`` so the
    caller can attach spectra to the correct compound.

    Notes
    -----
    The repository executes SQL only.  It does not commit the connection.
    """

    def __init__(self, connection, models):
        self.connection = connection
        self.table = models.ReferenceCompound.__table__

    def _unique_rows(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Collapse repeated structures while preserving first-seen metadata."""
        unique: dict[str, dict[str, Any]] = {}

        for row in rows:
            key = row.get("identity_key")
            if not isinstance(key, str) or not key:
                raise ContractError(
                    "Compound rows must contain a non-empty identity_key."
                )

            if key not in unique:
                unique[key] = _copy_compound_row(row)
            else:
                _merge_missing_metadata(unique[key], row)

        return unique

    def upsert_many(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[str, int]:
        """Insert or enrich compounds and return ``identity_key -> compound_id``.

        A new structure is inserted normally.  If its ``identity_key`` already
        exists, only missing optional metadata is filled.  Existing values have
        priority over newly imported values.

        ``extra_metadata`` follows the same rule.  PostgreSQL's JSONB ``||``
        operator keeps values from its right-hand object when keys overlap, so
        ``new_metadata || existing_metadata`` preserves existing database keys.
        """
        unique = self._unique_rows(rows)
        if not unique:
            return {}

        table = self.table
        values = list(unique.values())

        for block in chunks(values, COMPOUND_WRITE_BATCH_SIZE):
            statement = insert(table).values(block)

            updates = {
                column: func.coalesce(
                    table.c[column],
                    statement.excluded[column],
                )
                for column in FILLABLE_METADATA_COLUMNS
            }
            updates["extra_metadata"] = statement.excluded.extra_metadata.op(
                "||"
            )(table.c.extra_metadata)

            self.connection.execute(
                statement.on_conflict_do_update(
                    index_elements=[table.c.identity_key],
                    set_=updates,
                )
            )

        resolved: dict[str, int] = {}
        for block in chunks(list(unique), COMPOUND_ID_LOOKUP_BATCH_SIZE):
            statement = select(
                table.c.identity_key,
                table.c.id,
            ).where(table.c.identity_key.in_(block))

            for row in self.connection.execute(statement):
                resolved[row.identity_key] = int(row.id)

        missing = set(unique) - set(resolved)
        if missing:
            preview = sorted(missing)[:10]
            raise ContractError(
                "Compound upsert completed but IDs could not be resolved for "
                f"{len(missing)} structure(s): {preview}."
            )

        return resolved


class ReferenceSpectrumRepository:
    """Insert reference spectra and return their generated database IDs.

    The builder already validated each row before it reaches this class.  Rows
    are inserted in bounded blocks with ``INSERT ... RETURNING`` so PostgreSQL
    can return ``spectrum_id`` values without creating one SQLAlchemy ORM object
    per spectrum.

    Notes
    -----
    ``source_index`` is the stable physical record position in the source file.
    The returned mapping lets the import ledger connect that source record to
    the newly stored spectrum.
    """

    def __init__(self, connection, models):
        self.connection = connection
        self.table = models.ReferenceSpectrum.__table__

    def insert_many(
        self,
        rows: Iterable[Mapping[str, Any]],
    ) -> dict[int, int]:
        """Insert spectra and return ``source_index -> spectrum_id``."""
        resolved: dict[int, int] = {}
        expected: set[int] = set()

        for block in chunks(rows, SPECTRUM_WRITE_BATCH_SIZE):
            for input_row in block:
                source_index = int(input_row["source_index"])
                if source_index in expected:
                    raise ContractError(
                        "Spectrum rows contain the same source_index more than "
                        f"once: {source_index}."
                    )
                expected.add(source_index)

            statement = (
                insert(self.table)
                .values(block)
                .returning(
                    self.table.c.source_index,
                    self.table.c.id,
                )
            )

            for row in self.connection.execute(statement):
                source_index = int(row.source_index)
                if source_index in resolved:
                    raise ContractError(
                        "Spectrum insert returned the same source_index more "
                        f"than once: {source_index}."
                    )
                resolved[source_index] = int(row.id)

        if set(resolved) != expected:
            missing = sorted(expected - set(resolved))[:10]
            raise ContractError(
                "Spectrum insert did not return an ID for every source row; "
                f"missing source indices: {missing}."
            )

        return resolved


class ImportRepository:
    """Persist one completed import batch and its per-record decisions.

    ``BatchLedger`` tracks what happened to every physical source record in a
    batch.  Once the builder has finished preparing and storing the accepted
    records, this repository writes two things:

    * one ``import_batches`` row with accepted/rejected totals;
    * one ``import_records`` row for every source record, including rejection
      reasons and resolved compound/spectrum IDs.

    These rows are the durable checkpoint used when a build is resumed.
    Because this repository shares the builder's transaction, the checkpoint
    cannot commit without the corresponding compounds and spectra.
    """

    def __init__(self, connection, models):
        self.connection = connection
        self.models = models

    def finish(self, source_id: int, ledger) -> None:
        """Write a finished ledger as an atomic import checkpoint.

        The ledger must already contain exactly one decision for every record in
        the batch.  ``ledger.finish()`` is expected to enforce that invariant;
        the additional count check here protects the database-writing boundary
        against an inconsistent ledger implementation.
        """
        records = ledger.finish()
        consumed, accepted, rejected = ledger.counts()

        if len(records) != consumed:
            raise ContractError(
                "Import ledger row count does not match its consumed count: "
                f"{len(records)} != {consumed}."
            )

        batches = self.models.ImportBatch.__table__
        decisions = self.models.ImportRecord.__table__

        self.connection.execute(
            insert(batches).values(
                source_id=source_id,
                first_index=ledger.first,
                consumed=consumed,
                accepted=accepted,
                rejected=rejected,
            )
        )

        rows = [
            {
                "source_id": source_id,
                "source_index": decision.index,
                "batch_start": ledger.first,
                "status": "accepted" if decision.accepted else "rejected",
                "reason_code": decision.reason_code,
                "reason": decision.reason,
                "compound_id": decision.compound_id,
                "spectrum_id": decision.spectrum_id,
                "detail": decision.detail or {},
            }
            for decision in records
        ]

        for block in chunks(rows, IMPORT_RECORD_WRITE_BATCH_SIZE):
            self.connection.execute(insert(decisions).values(block))

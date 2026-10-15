"""Small record types used to keep import accounting exact and auditable.

The import pipeline treats every physical source row as one durable unit of work.
Each row starts as an :class:`InputRecord` and must end as exactly one
:class:`RecordDecision`: either accepted or rejected.

`BatchLedger` enforces that accounting rule for one contiguous source batch. It
is intentionally independent of database code so that parsing, validation, and
persistence cannot silently lose, duplicate, or classify a source row twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import ContractError, RejectedRecord


@dataclass(frozen=True)
class InputRecord:
    """One physical input row before scientific validation is complete.

    Parameters
    ----------
    index
        Zero-based position of the record in its source file. This physical
        position is preserved throughout parsing and import so that failures do
        not shift the identity of later records.
    value
        Parsed payload for the record. Depending on the source, this can be a
        metadata mapping, raw MGF text, or another intermediate representation.
    error
        Row-local parsing error detected before normal processing begins.

    Notes
    -----
    A record may contain either a usable ``value`` or an ``error``. Supplying
    both is ambiguous and therefore rejected.
    """

    index: int
    value: Any = None
    error: RejectedRecord | None = None

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("InputRecord.index must be a nonnegative integer.")

        if self.error is not None and not isinstance(self.error, RejectedRecord):
            raise TypeError("InputRecord.error must be a RejectedRecord or None.")

        if self.error is not None and self.value is not None:
            raise ValueError("InputRecord cannot contain both a value and an error.")


@dataclass(frozen=True)
class RecordDecision:
    """Final import decision for one physical source record.

    Accepted records must resolve to a database compound. Reference-spectrum
    records may additionally resolve to a spectrum ID. Rejected records carry a
    machine-readable ``reason_code`` and a human-readable ``reason``.

    The class itself performs only lightweight structural validation. The
    ledger remains responsible for checking that an index belongs to the
    current batch and that each source index is classified exactly once.
    """

    index: int
    accepted: bool
    reason_code: str | None = None
    reason: str | None = None
    compound_id: int | None = None
    spectrum_id: int | None = None
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.index, bool) or not isinstance(self.index, int) or self.index < 0:
            raise ValueError("RecordDecision.index must be a nonnegative integer.")

        if not isinstance(self.accepted, bool):
            raise TypeError("RecordDecision.accepted must be a boolean.")

        if not isinstance(self.detail, dict):
            raise TypeError("RecordDecision.detail must be a dictionary.")

        if self.accepted:
            if self.compound_id is None:
                raise ContractError("An accepted source record must resolve to a compound.")
            if self.reason_code is not None or self.reason is not None:
                raise ContractError("Accepted source records must not carry a rejection reason.")
        else:
            if not self.reason_code:
                raise ContractError("Rejected source records need a reason code.")
            if self.compound_id is not None or self.spectrum_id is not None:
                raise ContractError(
                    "Rejected source records must not resolve to compound or spectrum IDs."
                )


class BatchLedger:
    """Track one final decision for every source record in an import batch.

    Function of this class
    ----------------------
    Parsing and scientific validation can reject individual rows at different
    stages. Without one central ledger, it is easy for accepted/rejected counts
    to drift away from the actual source rows or for a record to be processed
    twice after error recovery.

    A ledger therefore enforces three simple rules:

    1. every decision belongs to this batch
    2. each source index is classified exactly once
    3. the batch is not considered complete until every source index has a
       decision.

    Parameters
    ----------
    first
        Source index of the first record in this batch.
    count
        Number of consecutive source records covered by the batch.
    """

    def __init__(self, first: int, count: int):
        if isinstance(first, bool) or not isinstance(first, int) or first < 0:
            raise ValueError("first must be a nonnegative integer.")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("count must be a nonnegative integer.")

        self.first = first
        self.count = count
        self._decisions: dict[int, RecordDecision] = {}

    @property
    def stop(self) -> int:
        """Exclusive upper source index covered by the batch."""
        return self.first + self.count

    def __len__(self) -> int:
        """Return the number of source rows already classified."""
        return len(self._decisions)

    def add(self, decision: RecordDecision) -> None:
        """Record one final decision for a source index.

        Raises
        ------
        ContractError
            If the source index falls outside the batch or was already
            classified.
        """
        if not isinstance(decision, RecordDecision):
            raise TypeError("decision must be a RecordDecision.")

        if not self.first <= decision.index < self.stop:
            raise ContractError(
                f"Source index {decision.index} is outside batch "
                f"[{self.first}, {self.stop})."
            )

        if decision.index in self._decisions:
            raise ContractError(f"Source index {decision.index} was classified twice.")

        self._decisions[decision.index] = decision

    def reject(self, index: int, error: RejectedRecord) -> None:
        """Record a row-local rejection using a :class:`RejectedRecord`."""
        if not isinstance(error, RejectedRecord):
            raise TypeError("error must be a RejectedRecord.")

        self.add(
            RecordDecision(
                index=index,
                accepted=False,
                reason_code=error.code,
                reason=str(error),
            )
        )

    def finish(self) -> list[RecordDecision]:
        """Return all decisions in source order after verifying completeness.

        This method does not mutate the ledger. It can safely be called more
        than once, for example once to persist decisions and again to report
        counts.
        """
        missing = [
            index
            for index in range(self.first, self.stop)
            if index not in self._decisions
        ]

        if missing:
            raise ContractError(
                "Unclassified source rows: "
                f"{missing[:10]}"
                + (" ..." if len(missing) > 10 else "")
            )

        return [self._decisions[index] for index in range(self.first, self.stop)]

    def counts(self) -> tuple[int, int, int]:
        """Return ``(consumed, accepted, rejected)`` for a complete batch."""
        decisions = self.finish()
        accepted = sum(decision.accepted for decision in decisions)
        rejected = self.count - accepted
        return self.count, accepted, rejected

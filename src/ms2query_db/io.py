"""Bounded, record-preserving input for MS2Query-DB builds.

The import layer has two responsibilities that are deliberately kept separate:

* discover immutable source files and describe their provenance;
* turn bounded source batches into row-aligned records without losing the
  physical source position of malformed entries.

MGF ingestion is collection-native. Raw MGF records are first isolated so that
one malformed record cannot shift the indices of all following records. Valid
sub-batches are then parsed with :func:`matchms.importing.load_ms2_dataset`,
which returns a :class:`matchms.SpectraCollection` directly. Parser failures are
bisected down to individual source records while successful collections are
concatenated without converting them back to individual ``Spectrum`` objects.
"""
from __future__ import annotations

import csv
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal, Self

from .errors import ContractError, RejectedRecord
from .records import InputRecord
from .util import file_sha256

if TYPE_CHECKING:
    from matchms import SpectraCollection


logger = logging.getLogger(__name__)

SourceKind = Literal["spectra", "chemicals"]
SOURCE_KINDS = frozenset({"spectra", "chemicals"})
_MGF_SUFFIX = ".mgf"
_MAX_MGF_PREAMBLE_BYTES = 1024 * 1024
_EXPECTED_PARSE_EXCEPTIONS = (ValueError, TypeError, AssertionError)


@dataclass(frozen=True)
class Source:
    """One immutable input file participating in a library build.

    Parameters
    ----------
    path
        Absolute path to the source file.
    sha256
        SHA-256 digest captured when the source is discovered.
    kind
        ``"spectra"`` for MGF reference spectra or ``"chemicals"`` for the
        optional compound metadata table.
    """

    path: Path
    sha256: str
    kind: SourceKind

    def __post_init__(self) -> None:
        path = Path(self.path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        if self.kind not in SOURCE_KINDS:
            raise ValueError(f"Unknown source kind {self.kind!r}; choose from {sorted(SOURCE_KINDS)}")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError("Source sha256 must be a 64-character hexadecimal digest")
        try:
            int(self.sha256, 16)
        except ValueError as exc:
            raise ValueError("Source sha256 must be hexadecimal") from exc
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "sha256", self.sha256.lower())

    @classmethod
    def from_path(cls, path: str | Path, kind: SourceKind) -> Self:
        """Create a source descriptor from a file and hash its current bytes."""
        resolved = Path(path).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        return cls(resolved, file_sha256(resolved), kind)

    def descriptor(self) -> dict[str, object]:
        """Return the JSON-compatible provenance stored with the library."""
        return {
            "name": self.path.name,
            "sha256": self.sha256,
            "kind": self.kind,
            "bytes": self.path.stat().st_size,
        }


@dataclass(frozen=True)
class LoadedSpectrumBatch:
    """Collection-native result of parsing one bounded MGF source batch.

    Attributes
    ----------
    collection
        Parsed spectra in source order. ``None`` when no record parsed
        successfully.
    source_indices
        Physical source-record indices aligned one-to-one with collection rows.
    rejected
        Mapping from physical source-record index to parse/record-boundary
        rejection. These indices are disjoint from ``source_indices``.

    Notes
    -----
    This object intentionally carries source indices separately instead of
    adding them to spectrum metadata. Source accounting is an MS2Query-DB
    concern and must not leak into scientific spectrum metadata or hashes.
    """

    collection: SpectraCollection | None
    source_indices: tuple[int, ...]
    rejected: dict[int, RejectedRecord]

    def __post_init__(self) -> None:
        if len(set(self.source_indices)) != len(self.source_indices):
            raise ContractError("Parsed MGF batch contains duplicate source indices")
        if set(self.source_indices) & set(self.rejected):
            raise ContractError("A source index cannot be both parsed and rejected")
        if self.collection is None:
            if self.source_indices:
                raise ContractError("Parsed source indices require a SpectraCollection")
        elif len(self.collection) != len(self.source_indices):
            raise ContractError(
                "SpectraCollection row count differs from parsed source-index count"
            )

    @property
    def accepted_count(self) -> int:
        """Number of successfully parsed spectra."""
        return len(self.source_indices)

    @property
    def rejected_count(self) -> int:
        """Number of records rejected before scientific validation."""
        return len(self.rejected)



def discover_sources(
    source: str | Path,
    *,
    recursive: bool = False,
    pattern: str = "*.mgf",
    kind: SourceKind = "spectra",
) -> list[Source]:
    """Discover deterministic, content-deduplicated build input files.

    Explicit file paths are used as-is. Directory inputs are searched with the
    supplied glob pattern, recursively when requested. Files with identical
    SHA-256 content are imported only once.

    Spectrum sources are deliberately restricted to MGF in this release. Other
    formats should be converted with matchms first so that record boundaries can
    be isolated and checkpointed reproducibly.
    """
    if kind not in SOURCE_KINDS:
        raise ValueError(f"Unknown source kind {kind!r}; choose from {sorted(SOURCE_KINDS)}")
    if not isinstance(pattern, str) or not pattern.strip():
        raise ValueError("pattern must be a non-empty glob pattern")
    if not isinstance(recursive, bool):
        raise TypeError("recursive must be a boolean")

    root = Path(source).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if root.is_file():
        paths = [root]
    elif root.is_dir():
        iterator = root.rglob(pattern) if recursive else root.glob(pattern)
        paths = sorted(path.resolve() for path in iterator if path.is_file())
    else:
        raise ValueError(f"Source path is neither a file nor directory: {root}")

    if not paths:
        raise ValueError(f"No files at {root} matching {pattern!r}")

    seen_hashes: dict[str, Path] = {}
    result: list[Source] = []
    for path in paths:
        if kind == "spectra" and path.suffix.lower() != _MGF_SUFFIX:
            raise ValueError(
                "MS2Query-DB currently uses record-isolated MGF ingestion for "
                f"reference spectra; got {path.name!r}. Convert other formats "
                "with matchms before building the library."
            )

        item = Source.from_path(path, kind)
        previous = seen_hashes.get(item.sha256)
        if previous is not None:
            logger.info(
                "Skipping duplicate source content %s; identical to %s",
                item.path,
                previous,
            )
            continue

        seen_hashes[item.sha256] = item.path
        result.append(item)

    return result



def _mgf_records(path: Path) -> Iterator[InputRecord]:
    """Yield isolated raw MGF records with stable physical record indices.

    Malformation inside a record becomes a row-level ``RejectedRecord`` where
    alignment can still be determined. File-level structural ambiguity (for
    example an unmatched ``END IONS`` outside a record) raises immediately,
    because silently guessing subsequent record positions would corrupt resume
    accounting.
    """
    block: list[str] | None = None
    preamble: list[str] = []
    preamble_bytes = 0
    seen_record = False
    record_index = 0

    with path.open(encoding="utf-8-sig") as stream:
        for line_number, line in enumerate(stream, start=1):
            token = line.strip().upper()

            if token == "BEGIN IONS":
                if block is not None:
                    yield InputRecord(
                        record_index,
                        error=RejectedRecord(
                            "malformed_mgf",
                            "Nested BEGIN IONS before END IONS",
                        ),
                    )
                    record_index += 1

                block = [line]
                seen_record = True
                continue

            if token == "END IONS":
                if block is None:
                    raise ValueError(
                        f"MGF {path} contains END IONS without BEGIN IONS "
                        f"at line {line_number}; repair record boundaries before importing"
                    )

                block.append(line if line.endswith("\n") else line + "\n")
                yield InputRecord(
                    record_index,
                    "".join((*preamble, *block)),
                )
                record_index += 1
                block = None
                continue

            if block is not None:
                block.append(line)
                continue

            if not seen_record:
                preamble.append(line)
                preamble_bytes += len(line.encode("utf-8"))
                if preamble_bytes > _MAX_MGF_PREAMBLE_BYTES:
                    raise ValueError(
                        f"MGF preamble exceeds {_MAX_MGF_PREAMBLE_BYTES} bytes: {path}"
                    )
                continue

            if token and not token.startswith(("#", ";", "!")):
                raise ValueError(
                    f"Unexpected data outside an MGF record at {path}:{line_number}; "
                    "source alignment is ambiguous"
                )

    if block is not None:
        yield InputRecord(
            record_index,
            error=RejectedRecord(
                "malformed_mgf",
                "Missing final END IONS",
            ),
        )
    elif record_index == 0:
        raise ValueError(f"No MGF records found in {path}")



def _csv_records(
    path: Path,
    delimiter: str,
    smiles_column: str,
) -> Iterator[InputRecord]:
    """Yield normalized CSV rows while retaining their data-row indices."""
    if not isinstance(delimiter, str) or len(delimiter) != 1:
        raise ValueError("CSV delimiter must contain exactly one character")
    if not isinstance(smiles_column, str) or not smiles_column.strip():
        raise ValueError("smiles_column must be a non-empty string")

    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=delimiter)
        raw_names = reader.fieldnames or []
        names = [str(name or "").strip().lower() for name in raw_names]

        if not names or "" in names or len(set(names)) != len(names):
            raise ValueError(
                "CSV requires nonempty, case-insensitively unique column names"
            )

        smiles_key = smiles_column.strip().lower()
        if smiles_key not in names:
            raise ValueError(
                f"CSV lacks configured SMILES column {smiles_column!r}"
            )

        # Normalize keys once instead of rebuilding/stripping them for every row.
        reader.fieldnames = names

        for index, row in enumerate(reader):
            if None in row:
                yield InputRecord(
                    index,
                    error=RejectedRecord(
                        "malformed_csv",
                        "More fields than header columns",
                    ),
                )
                continue

            yield InputRecord(index, dict(row))



def iter_source_batches(
    source: Source,
    batch_size: int,
    *,
    delimiter: str = ",",
    smiles_column: str = "smiles",
) -> Iterator[list[InputRecord]]:
    """Yield bounded, consecutive record batches from one build source."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")

    if source.kind == "spectra":
        iterator = _mgf_records(source.path)
    elif source.kind == "chemicals":
        iterator = _csv_records(
            source.path,
            delimiter,
            smiles_column,
        )
    else:  # Defensive even though Source validates this already.
        raise ValueError(f"Unsupported source kind {source.kind!r}")

    block: list[InputRecord] = []
    for record in iterator:
        block.append(record)
        if len(block) == batch_size:
            yield block
            block = []

    if block:
        yield block



def load_mgf_batch(
    records: Sequence[InputRecord],
    *,
    mz_precision: float = 1e-6,
    loader=None,
) -> LoadedSpectrumBatch:
    """Parse isolated MGF records directly into a row-aligned collection.

    Parameters
    ----------
    records
        One bounded sequence emitted by :func:`iter_source_batches`. Records
        already rejected by record-boundary parsing remain rejected without
        being sent through matchms.
    mz_precision
        m/z grid precision passed to :class:`matchms.SpectraCollection` through
        :func:`matchms.importing.load_ms2_dataset`.
    loader
        Optional replacement for ``load_ms2_dataset`` used by unit tests.

    Returns
    -------
    LoadedSpectrumBatch
        Successful spectra remain in one ``SpectraCollection`` and are aligned
        to ``source_indices``. Parser failures are isolated recursively and
        returned in ``rejected``.

    Notes
    -----
    Normal valid batches require a single matchms parse call. Recursive bisection
    is used only after a parser error or row-count mismatch. This protects
    physical source indices without paying per-spectrum parsing overhead in the
    normal case.
    """
    records = tuple(records)
    if not records:
        return LoadedSpectrumBatch(None, (), {})

    input_indices = [record.index for record in records]
    if len(set(input_indices)) != len(input_indices):
        raise ContractError("MGF batch contains duplicate source indices")

    if loader is None:
        from matchms.importing import load_ms2_dataset

        loader = load_ms2_dataset

    rejected: dict[int, RejectedRecord] = {
        record.index: record.error
        for record in records
        if record.error is not None
    }
    pending = [record for record in records if record.error is None]

    parsed_chunks: list[tuple[tuple[int, ...], object]] = []

    with TemporaryDirectory(prefix="ms2query_parse_") as directory:
        path = Path(directory) / "batch.mgf"

        def parse(part: Sequence[InputRecord]) -> None:
            if not part:
                return

            path.write_text(
                "".join(str(record.value) for record in part),
                encoding="utf-8",
            )

            try:
                collection = loader(
                    str(path),
                    metadata_harmonization=True,
                    ftype="mgf",
                    mz_precision=mz_precision,
                )
                if collection is None:
                    raise ValueError("Parser returned no SpectraCollection")
                if len(collection) != len(part):
                    raise ValueError(
                        f"Parser returned {len(collection)} spectra for "
                        f"{len(part)} MGF records"
                    )
            except _EXPECTED_PARSE_EXCEPTIONS as exc:
                if len(part) > 1:
                    middle = len(part) // 2
                    parse(part[:middle])
                    parse(part[middle:])
                else:
                    record = part[0]
                    rejected[record.index] = RejectedRecord(
                        "spectrum_parse",
                        str(exc),
                    )
                return

            parsed_chunks.append(
                (tuple(record.index for record in part), collection)
            )

        parse(pending)

    parsed_chunks.sort(key=lambda item: item[0][0] if item[0] else -1)
    source_indices = tuple(
        index
        for indices, _collection in parsed_chunks
        for index in indices
    )

    if not parsed_chunks:
        collection = None
    elif len(parsed_chunks) == 1:
        collection = parsed_chunks[0][1]
    else:
        from matchms import SpectraCollection

        try:
            collection = SpectraCollection.concat(
                [chunk for _indices, chunk in parsed_chunks]
            )
        except (ValueError, TypeError, AssertionError) as exc:
            raise ContractError(
                "Could not concatenate successfully parsed MGF sub-batches"
            ) from exc

    result = LoadedSpectrumBatch(
        collection=collection,
        source_indices=source_indices,
        rejected=rejected,
    )

    accounted = set(result.source_indices) | set(result.rejected)
    expected = set(input_indices)
    if accounted != expected:
        missing = sorted(expected - accounted)
        unexpected = sorted(accounted - expected)
        raise ContractError(
            "MGF parser lost or invented source indices; "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )

    return result

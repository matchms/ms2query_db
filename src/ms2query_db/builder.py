"""Build MS2Query-DB libraries from spectra and compound metadata.

The builder owns the import state machine and batch transaction boundaries. Each
physical source record is classified exactly once as accepted or rejected via
:class:`BatchLedger`; there are no parallel "accepted index" arrays that can
silently lose alignment.

Reference spectra are handled collection-first. MGF parsing still preserves the
physical record index of every source entry, but validation and MS2DeepScore
inference operate on :class:`matchms.SpectraCollection` batches. For a library
configured with ``ionmode="both"``, the actual positive/negative mode of every
accepted reference spectrum is stored explicitly.
"""
from __future__ import annotations

import hashlib
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chemistry import (
    ChemapFingerprinter,
    FingerprintProjector,
    encode_fingerprint,
    identify_compound,
)
from .embeddings import validated_embeddings
from .errors import CompatibilityError, ContractError, RejectedRecord
from .io import Source, discover_sources, iter_source_batches, load_mgf_batch
from .processing import strip_structure_annotations, validate_collection
from .records import BatchLedger
from .util import file_sha256, jsonable, scientific_versions, text_value

logger = logging.getLogger(__name__)

BUILD_PIPELINE = "ms2query-db/0.1"  # TODO: check versioning

IDENTIFIER_KEYS = {
    "smiles",
    "inchi",
    "inchikey",
    "inchikey14",
    "formula",
    "mass",
    "parent_mass",
    "exact_mass",
}

_TYPED_COMPOUND_METADATA = {
    "superclass",
    "class",
    "compound_class",
    "subclass",
    "visual_x",
    "visual_y",
    "pubchem_cid",
}


@dataclass(frozen=True)
class BuildReport:
    """Compact summary returned after a library build or completed resume."""

    schema: str
    library_id: str
    counts: dict
    resumed_batches: int


def check_metadata(value: object) -> None:
    """Reject text that cannot safely be represented in PostgreSQL JSON/text.

    The check is deliberately recursive because arbitrary compound/source
    metadata is retained in JSONB. PostgreSQL text cannot contain NUL bytes and
    invalid Unicode must fail before an otherwise valid batch reaches the
    database transaction.
    """
    if isinstance(value, str):
        if "\x00" in value:
            raise RejectedRecord(
                "invalid_metadata",
                "Text contains a NUL character",
            )
        try:
            value.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise RejectedRecord(
                "invalid_metadata",
                "Text contains invalid Unicode",
            ) from exc
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            check_metadata(key)
            check_metadata(item)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            check_metadata(item)


def compound_metadata(row: Mapping[str, object]) -> dict:
    """Split compound metadata into typed columns and lossless JSON extras.

    Known taxonomy fields and optional numeric fields are stored in dedicated
    PostgreSQL columns. Unknown columns remain in ``extra_metadata``. Invalid
    optional numeric values do not reject an otherwise usable molecule; their
    original values are retained in ``extra_metadata`` and a warning is logged.
    """
    normalized = {
        str(key).strip().lower(): jsonable(value)
        for key, value in row.items()
    }
    check_metadata(normalized)

    result = {
        key: text_value(normalized.get(key)) or None
        for key in ("superclass", "subclass")
    }
    result["compound_class"] = (
        text_value(
            normalized.get(
                "class",
                normalized.get("compound_class"),
            )
        )
        or None
    )

    invalid_optional: set[str] = set()
    for key in ("visual_x", "visual_y", "pubchem_cid"):
        raw = normalized.get(key)
        value = text_value(raw)
        result[key] = None
        if not value:
            continue

        try:
            number = float(value)
            if not np.isfinite(number):
                raise ValueError("nonfinite")

            if key == "pubchem_cid":
                if number != int(number) or not 0 <= number <= 2**63 - 1:
                    raise ValueError("invalid integer")
                number = int(number)

            result[key] = number
        except (ValueError, TypeError, OverflowError):
            invalid_optional.add(key)
            logger.warning(
                "Ignoring invalid optional %s=%r; raw value retained in "
                "extra_metadata",
                key,
                raw,
            )

    excluded = IDENTIFIER_KEYS | _TYPED_COMPOUND_METADATA
    extras = {
        key: value
        for key, value in normalized.items()
        if key not in excluded
        and value is not None
        and text_value(value)
    }
    for key in invalid_optional:
        extras[key] = normalized[key]

    result["extra_metadata"] = extras
    return result


class LibraryBuilder:
    """Build a resumable MS2Query-DB reference library.

    Parameters
    ----------
    database
        Database service owning schema creation, transactional batch commits,
        and final PostgreSQL indexes.
    config
        :class:`BuildConfig` describing the immutable scientific/library
        representation.
    embedder
        Optional MS2DeepScore embedding adapter. If supplied, every accepted
        reference spectrum must also produce a valid embedding.
    fingerprinter
        Optional chemical fingerprinter. The default uses the configured
        Chemap-based implementation.
    spectrum_loader
        Record-preserving MGF loader. Primarily injectable for tests.
    spectrum_stripper
        Function removing structural annotations before model inference. It
        must accept a :class:`matchms.SpectraCollection`.

    Notes
    -----
    The builder does not preprocess/clean reference peaks with the query
    processing pipeline. Parsed reference peaks are treated as canonical input
    and are validated only. And for now, query preprocessing remains a separate concern.
    """

    def __init__(
        self,
        database,
        config,
        *,
        embedder=None,
        fingerprinter=None,
        spectrum_loader=load_mgf_batch,
        spectrum_stripper=strip_structure_annotations,
    ):
        self.database = database
        self.config = config
        self.embedder = embedder
        self.fingerprinter = fingerprinter or ChemapFingerprinter(
            config.fingerprints
        )
        self.spectrum_loader = spectrum_loader
        self.spectrum_stripper = spectrum_stripper
        self.projector = FingerprintProjector.create(config.fingerprints)
        self.resumed_batches = 0

    def _reject(
        self,
        ledger: BatchLedger,
        index: int,
        error: RejectedRecord,
    ) -> None:
        """Reject one source row according to the configured invalid policy."""
        if self.config.invalid_policy == "raise":
            raise RejectedRecord(
                error.code,
                f"Source record {index}: {error}",
            ) from error
        ledger.reject(index, error)

    @staticmethod
    def _require_contiguous_batch(records: Sequence, ledger: BatchLedger) -> None:
        indices = [record.index for record in records]
        expected = list(range(ledger.first, ledger.first + ledger.count))
        if indices != expected:
            raise ContractError("Input batch indices must be contiguous")

    def _prepare_reference_records(
        self,
        records,
        ledger: BatchLedger,
    ) -> tuple[list[dict], object | None]:
        """Parse and validate reference spectra without leaving collection form."""
        cfg = self.config
        loaded = self.spectrum_loader(
            records,
            mz_precision=cfg.mz_precision,
        )

        for source_index, error in loaded.rejected.items():
            self._reject(ledger, source_index, error)

        collection = loaded.collection
        if collection is None:
            return [], None

        source_indices = loaded.source_indices
        if len(collection) != len(source_indices):
            raise ContractError(
                "Parsed reference collection lost source-index alignment"
            )

        errors, precursors, ionmodes = validate_collection(
            collection,
            cfg.ionmode,
        )
        if len(errors) != len(collection):
            raise ContractError(
                "Reference validation changed collection row alignment"
            )

        metadata_table = collection.metadata
        fragment_hashes = collection.fragment_hashes
        if len(fragment_hashes) != len(collection):
            raise ContractError(
                "Reference fragment hashes changed collection row alignment"
            )

        pending: list[dict] = []
        for collection_row, (
            source_index,
            validation_error,
            precursor,
            ionmode,
        ) in enumerate(
            zip(
                source_indices,
                errors,
                precursors,
                ionmodes,
                strict=True,
            )
        ):
            if validation_error is not None:
                self._reject(
                    ledger,
                    source_index,
                    validation_error,
                )
                continue

            try:
                metadata = jsonable(
                    metadata_table.iloc[collection_row].to_dict()
                )
                check_metadata(metadata)

                identity = identify_compound(
                    text_value(metadata.get("smiles")),
                    text_value(metadata.get("inchi")),
                    text_value(metadata.get("inchikey")),
                )

                retention = metadata.get("retention_time")
                try:
                    retention = (
                        float(retention)
                        if retention is not None
                        else None
                    )
                    if retention is not None and not np.isfinite(retention):
                        retention = None
                except (ValueError, TypeError):
                    retention = None

                mz, intensity = collection.fragments.get_row(collection_row)
                mz = np.asarray(mz, dtype=np.float64)
                intensity = np.asarray(intensity, dtype=np.float64)

                spectrum_row = {
                    "n_peaks": len(mz),
                    "mz_values": mz.astype("<f8", copy=False).tobytes(),
                    "intensity_values": intensity.astype(
                        "<f8",
                        copy=False,
                    ).tobytes(),
                    "spectrum_hash": str(fragment_hashes[collection_row]),
                    "precursor_mz": float(precursor),
                    "ionmode": ionmode,
                    "retention_time": retention,
                    "other_metadata": metadata,
                    "embedding": None,
                }

                typed_metadata = compound_metadata(
                    {
                        key: value
                        for key, value in metadata.items()
                        if key in _TYPED_COMPOUND_METADATA
                    }
                )
            except RejectedRecord as exc:
                self._reject(ledger, source_index, exc)
                continue

            pending.append(
                {
                    "index": source_index,
                    "identity": identity,
                    "metadata": typed_metadata,
                    "spectrum": spectrum_row,
                    "collection_row": collection_row,
                    "detail": {"ionmode": ionmode},
                }
            )

        return pending, collection

    def _prepare_compound_records(
        self,
        records,
        ledger: BatchLedger,
    ) -> list[dict]:
        """Prepare chemical metadata rows without creating spectrum records."""
        cfg = self.config
        smiles_key = cfg.smiles_column.lower().strip()
        inchi_key = (
            cfg.inchi_column.lower().strip()
            if cfg.inchi_column
            else None
        )
        inchikey_key = (
            cfg.inchikey_column.lower().strip()
            if cfg.inchikey_column
            else None
        )

        pending = []
        for record in records:
            if record.error is not None:
                self._reject(ledger, record.index, record.error)
                continue

            try:
                metadata = record.value
                if not isinstance(metadata, Mapping):
                    raise RejectedRecord(
                        "invalid_metadata",
                        "Chemical input record must be a mapping",
                    )

                identity = identify_compound(
                    text_value(metadata.get(smiles_key)),
                    text_value(metadata.get(inchi_key)) if inchi_key else "",
                    text_value(metadata.get(inchikey_key))
                    if inchikey_key
                    else "",
                )
                typed_metadata = compound_metadata(metadata)
            except RejectedRecord as exc:
                self._reject(ledger, record.index, exc)
                continue

            pending.append(
                {
                    "index": record.index,
                    "identity": identity,
                    "metadata": typed_metadata,
                    "spectrum": None,
                    "collection_row": None,
                    "detail": {},
                }
            )

        return pending

    def _attach_fingerprints(
        self,
        pending: list[dict],
        ledger: BatchLedger,
    ) -> list[dict]:
        """Compute each unique molecular fingerprint once per source batch."""
        unique = {
            row["identity"].identity_key: row["identity"]
            for row in pending
        }
        if not unique:
            return []

        fingerprints = self.fingerprinter.compute_many(
            [identity.smiles for identity in unique.values()]
        )
        if len(fingerprints) != len(unique):
            raise ContractError("Fingerprinter changed row alignment")

        fp_map = dict(zip(unique, fingerprints, strict=True))
        accepted = []
        for row in pending:
            fingerprint = fp_map[row["identity"].identity_key]
            if isinstance(fingerprint, RejectedRecord):
                self._reject(
                    ledger,
                    row["index"],
                    fingerprint,
                )
                continue

            row["fingerprint"] = fingerprint
            accepted.append(row)

        return accepted

    def _attach_embeddings(
        self,
        accepted: list[dict],
        collection,
        ledger: BatchLedger,
    ) -> list[dict]:
        """Embed accepted reference spectra as one aligned collection batch."""
        if self.embedder is None or not accepted:
            return accepted
        if collection is None:
            raise ContractError(
                "Reference embeddings requested without a reference collection"
            )

        rows = np.asarray(
            [row["collection_row"] for row in accepted],
            dtype=np.int64,
        )
        inference_collection = collection[rows]
        inference_collection = self.spectrum_stripper(
            inference_collection
        )

        embeddings = validated_embeddings(
            self.embedder,
            inference_collection,
        )
        if len(embeddings) != len(accepted):
            raise ContractError("Embedder changed reference row alignment")

        valid = []
        for row, embedding in zip(
            accepted,
            embeddings,
            strict=True,
        ):
            if isinstance(embedding, RejectedRecord):
                self._reject(
                    ledger,
                    row["index"],
                    embedding,
                )
                continue

            row["spectrum"]["embedding"] = embedding
            valid.append(row)

        return valid

    def _materialize_compound_rows(
        self,
        accepted: list[dict],
    ) -> dict[str, dict]:
        """Create encoded compound rows and deterministic ANN projections."""
        unique_valid: dict[str, dict] = {}
        for row in accepted:
            unique_valid.setdefault(
                row["identity"].identity_key,
                row,
            )

        if not unique_valid:
            return {}

        fingerprints = np.stack(
            [row["fingerprint"] for row in unique_valid.values()]
        )
        projections = self.projector.transform(fingerprints)
        if len(projections) != len(unique_valid):
            raise ContractError(
                "Fingerprint projector changed compound row alignment"
            )

        compound_rows = {}
        for row, projection in zip(
            unique_valid.values(),
            projections,
            strict=True,
        ):
            identity = row["identity"]
            compound_rows[identity.identity_key] = dict(
                identity.to_dict(),
                fingerprint=encode_fingerprint(row["fingerprint"]),
                fingerprint_projection=projection,
            )

        return compound_rows

    def _prepare(self, records, *, reference: bool):
        """Classify every source row exactly once and return valid DB payloads."""
        if not records:
            raise ContractError("Cannot prepare an empty source batch")

        ledger = BatchLedger(records[0].index, len(records))
        self._require_contiguous_batch(records, ledger)

        if reference:
            pending, collection = self._prepare_reference_records(
                records,
                ledger,
            )
        else:
            pending = self._prepare_compound_records(
                records,
                ledger,
            )
            collection = None

        accepted = self._attach_fingerprints(
            pending,
            ledger,
        )
        if reference:
            accepted = self._attach_embeddings(
                accepted,
                collection,
                ledger,
            )

        # Only fully valid rows may cause compound creation. This is important
        # for references: an invalid embedding must not leave behind an orphaned
        # compound merely because its structure/fingerprint was valid.
        compound_rows = self._materialize_compound_rows(accepted)

        prepared = []
        for row in accepted:
            key = row["identity"].identity_key
            prepared.append(
                {
                    "index": row["index"],
                    "compound": dict(
                        compound_rows[key],
                        **row["metadata"],
                    ),
                    "spectrum": row["spectrum"],
                    "detail": row["detail"],
                }
            )

        return ledger, prepared

    def _sources(self, source, chemical_csv) -> list[Source]:
        """Resolve immutable source descriptors in deterministic import order."""
        cfg = self.config
        references = discover_sources(
            source,
            recursive=cfg.recursive,
            pattern=cfg.pattern,
        )

        chemicals = []
        if chemical_csv is not None:
            path = Path(chemical_csv).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            chemicals = [
                Source(
                    path,
                    file_sha256(path),
                    "chemicals",
                )
            ]

        # Compound metadata is imported first so later spectra can fill/use the
        # same structure rows without a second scientific representation.
        return chemicals + references

    def _descriptor(self, sources: Sequence[Source]) -> dict:
        """Return the immutable scientific build descriptor used for resume."""
        model = None
        if self.embedder is not None:
            model = {
                "model_id": self.embedder.model_id,
                "dimension": self.embedder.dimension,
            }

        return {
            "config": self.config.to_dict(),
            "sources": [source.descriptor() for source in sources],
            "model": model,
            "packages": scientific_versions(),
            "pipeline": BUILD_PIPELINE,
        }

    def build(
        self,
        source,
        chemical_csv=None,
        *,
        resume: bool = False,
    ) -> BuildReport:
        """Build or resume a library and return its final import statistics.

        Resume is deliberately strict: configuration, source hashes, package
        versions, embedding model, and projection must describe the identical
        scientific build. Completed batch checkpoints are reused atomically.
        """
        cfg = self.config
        sources = self._sources(source, chemical_csv)
        descriptor = self._descriptor(sources)

        with self.database.build_lock():
            info = self.database.initialize(
                cfg,
                descriptor,
                self.projector,
                resume=resume,
            )

            payload = bytes(info["projection"])
            if (
                hashlib.sha256(payload).hexdigest()
                != info["projection_sha256"]
            ):
                raise CompatibilityError(
                    "Stored fingerprint projection checksum mismatch"
                )

            self.projector = FingerprintProjector.from_bytes(
                payload,
                cfg.fingerprints,
            )

            if info["state"] == "ready":
                return BuildReport(
                    cfg.schema,
                    info["library_id"],
                    self.database.statistics(),
                    0,
                )

            self.database.set_state("importing")

            try:
                for source_descriptor in sources:
                    source_id = self.database.source_id(
                        source_descriptor
                    )
                    completed = self.database.completed_batches(
                        source_id
                    )

                    for records in iter_source_batches(
                        source_descriptor,
                        cfg.batch_size,
                        delimiter=cfg.csv_delimiter,
                        smiles_column=cfg.smiles_column,
                    ):
                        first = records[0].index

                        if first in completed:
                            if completed[first] != len(records):
                                raise CompatibilityError(
                                    "Resume batch boundaries changed"
                                )
                            self.resumed_batches += 1
                            continue

                        reference = source_descriptor.kind == "spectra"
                        ledger, prepared = self._prepare(
                            records,
                            reference=reference,
                        )
                        self.database.commit_batch(
                            source_id,
                            ledger,
                            prepared,
                            reference=reference,
                        )

                        consumed, accepted, rejected = ledger.counts()
                        logger.info(
                            "Committed %s %s records %d..%d: %d accepted, "
                            "%d rejected",
                            source_descriptor.kind,
                            source_descriptor.path.name,
                            first,
                            first + consumed - 1,
                            accepted,
                            rejected,
                        )

                    if (
                        file_sha256(source_descriptor.path)
                        != source_descriptor.sha256
                    ):
                        raise CompatibilityError(
                            "Source file changed during import; do not resume "
                            "with changed input"
                        )

                self.database.finalize()

            except BaseException as exc:
                try:
                    self.database.set_state(
                        "failed",
                        f"{type(exc).__name__}: {str(exc)[:2000]}",
                    )
                except Exception:
                    logger.exception(
                        "Could not record failed state; committed checkpoints "
                        "still determine resumption"
                    )
                raise

            return BuildReport(
                cfg.schema,
                info["library_id"],
                self.database.statistics(),
                self.resumed_batches,
            )

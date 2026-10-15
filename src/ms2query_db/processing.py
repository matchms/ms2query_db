"""Collection-oriented spectrum preprocessing and validation.

Query spectra are processed in batches using :class:`matchms.SpectraCollection`.
Reference spectra already stored by MS2Query-DB are treated as canonical and are
never re-cleaned when reconstructed for scoring.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .errors import ContractError, RejectedRecord
from .util import text_value

if TYPE_CHECKING:
    from matchms import SpectraCollection, SpectraProcessor, Spectrum

    from .config import ProcessingSpec


STRUCTURE_FIELDS = frozenset(
    {
        "smiles",
        "inchi",
        "inchikey",
        "inchikey14",
        "formula",
        "fingerprint",
        "parent_mass",
        "exact_mass",
        "compound_name",
        "superclass",
        "class",
        "subclass",
        "visual_x",
        "visual_y",
    }
)

_QUERY_ROW_ID = "ms2query_input_row"
_VALID_IONMODES = frozenset({"positive", "negative"})


@dataclass(frozen=True)
class ProcessedQueryCollection:
    """Result of collection-level query preprocessing.

    Attributes
    ----------
    spectra
        Processed query spectra. ``None`` means that no input spectrum survived
        preprocessing and validation.
    input_rows
        Row numbers in the input sequence corresponding to rows in ``spectra``.
        This preserves alignment even when matchms filters remove spectra.
    precursor_mz
        Validated precursor m/z values aligned to ``spectra``.
    ionmodes
        Canonical ``"positive"``/``"negative"`` ion modes aligned to
        ``spectra``.
    rejected
        Mapping from input row number to the reason that row was rejected.
    """

    spectra: SpectraCollection | None
    input_rows: tuple[int, ...]
    precursor_mz: np.ndarray
    ionmodes: tuple[str, ...]
    rejected: dict[int, RejectedRecord]


def make_processor(spec: ProcessingSpec) -> SpectraProcessor:
    """Create the standard query-spectrum preprocessing pipeline.

    The returned :class:`matchms.SpectraProcessor` supports both single spectra
    and :class:`matchms.SpectraCollection` objects. MS2Query-DB uses its
    collection path for query batches.
    """
    from matchms import SpectraProcessor
    from matchms.filtering import (
        default_filters,
        normalize_intensities,
        reduce_to_number_of_peaks,
        select_by_mz,
        select_by_relative_intensity,
    )

    return SpectraProcessor(
        filters=[
            default_filters,
            (
                select_by_mz,
                {"mz_from": spec.mz_min, "mz_to": spec.mz_max},
            ),
            normalize_intensities,
            (
                select_by_relative_intensity,
                {"intensity_from": spec.min_relative_intensity},
            ),
            (
                reduce_to_number_of_peaks,
                {
                    "n_required": spec.min_peaks,
                    "n_max": spec.max_peaks,
                },
            ),
        ]
    )


def strip_structure_annotations(spectra):
    """Return spectra without structural annotations used only for evaluation.

    Structural metadata must never become an accidental model or search feature.
    The caller's object is not mutated. Both :class:`matchms.Spectrum` and
    :class:`matchms.SpectraCollection` inputs are supported so that reference
    building can remain spectrum-oriented until it is refactored separately.
    """
    from matchms import SpectraCollection, Spectrum

    if isinstance(spectra, SpectraCollection):
        columns = [
            column
            for column in spectra.metadata.columns
            if str(column).lower() in STRUCTURE_FIELDS
        ]
        if not columns:
            return spectra.copy()
        return spectra.drop_metadata(columns, errors="ignore")

    if isinstance(spectra, Spectrum):
        metadata = {
            key: value
            for key, value in dict(spectra.metadata).items()
            if str(key).lower() not in STRUCTURE_FIELDS
        }
        return Spectrum(
            mz=np.asarray(spectra.peaks.mz, dtype=np.float64).copy(),
            intensities=np.asarray(spectra.peaks.intensities, dtype=np.float64).copy(),
            metadata=metadata,
            metadata_harmonization=False,
        )

    raise TypeError(
        "strip_structure_annotations expects a matchms Spectrum or "
        "SpectraCollection."
    )


def validate_ionmode(value, library_ionmode: str) -> str:
    """Return a canonical spectrum ion mode compatible with the library."""
    mode = text_value(value).lower()

    if mode not in _VALID_IONMODES:
        raise RejectedRecord(
            "invalid_ionmode",
            "ionmode must be 'positive' or 'negative'",
        )

    if library_ionmode not in {*_VALID_IONMODES, "both"}:
        raise ValueError(
            "library_ionmode must be 'positive', 'negative', or 'both'"
        )

    if library_ionmode != "both" and mode != library_ionmode:
        raise RejectedRecord(
            "ionmode_mismatch",
            f"Expected {library_ionmode}; got {mode!r}",
        )

    return mode


def validate_spectrum(spectrum: Spectrum, library_ionmode: str):
    """Validate one spectrum and return canonical peak/precursor arrays.

    This single-spectrum helper remains for the reference builder until the
    builder itself is migrated to ``SpectraCollection``. Query batches use
    :func:`process_query_collection` instead.
    """
    if spectrum is None:
        raise RejectedRecord(
            "filtered_spectrum",
            "Processor returned no spectrum",
        )

    mz = np.asarray(spectrum.peaks.mz, dtype=np.float64)
    intensity = np.asarray(spectrum.peaks.intensities, dtype=np.float64)

    if mz.ndim != 1 or mz.shape != intensity.shape or not mz.size:
        raise RejectedRecord(
            "empty_spectrum",
            "Peak arrays must be equally sized and nonempty",
        )

    if not np.isfinite(mz).all() or not np.isfinite(intensity).all():
        raise RejectedRecord(
            "invalid_peaks",
            "Peak arrays contain nonfinite values",
        )

    if (
        np.any(mz <= 0)
        or np.any(np.diff(mz) < 0)
        or np.any(intensity < 0)
        or not np.any(intensity > 0)
    ):
        raise RejectedRecord(
            "invalid_peaks",
            "Require sorted positive masses and nonnegative, nonzero intensities",
        )

    try:
        precursor = float(spectrum.get("precursor_mz"))
    except (ValueError, TypeError) as exc:
        raise RejectedRecord(
            "invalid_precursor",
            "Missing or nonnumeric precursor_mz",
        ) from exc

    if not np.isfinite(precursor) or precursor <= 0:
        raise RejectedRecord(
            "invalid_precursor",
            "precursor_mz must be finite and positive",
        )

    validate_ionmode(spectrum.get("ionmode"), library_ionmode)

    # Canonical storage uses float64 for both arrays; intensities are never
    # silently narrowed or truncated.
    return mz, intensity, precursor


def validate_collection(
    collection: SpectraCollection,
    library_ionmode: str,
) -> tuple[list[RejectedRecord | None], np.ndarray, tuple[str, ...]]:
    """Validate processed collection rows without reconstructing Spectrum objects.

    Validation is row-aligned: one malformed query does not invalidate the
    remainder of the batch.

    Returns
    -------
    errors
        One ``RejectedRecord`` or ``None`` per collection row.
    precursor_mz
        Parsed precursor values. Rejected rows contain ``NaN``.
    ionmodes
        Canonical ion-mode strings aligned to collection rows.
    """
    errors: list[RejectedRecord | None] = []
    precursors = np.full(len(collection), np.nan, dtype=np.float64)
    ionmodes: list[str] = []
    metadata = collection.metadata

    for row_index in range(len(collection)):
        error: RejectedRecord | None = None
        mz, intensity = collection.fragments.get_row(row_index)
        mz = np.asarray(mz, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)

        if mz.ndim != 1 or mz.shape != intensity.shape or not mz.size:
            error = RejectedRecord(
                "empty_spectrum",
                "Peak arrays must be equally sized and nonempty",
            )
        elif not np.isfinite(mz).all() or not np.isfinite(intensity).all():
            error = RejectedRecord(
                "invalid_peaks",
                "Peak arrays contain nonfinite values",
            )
        elif (
            np.any(mz <= 0)
            or np.any(np.diff(mz) < 0)
            or np.any(intensity < 0)
            or not np.any(intensity > 0)
        ):
            error = RejectedRecord(
                "invalid_peaks",
                "Require sorted positive masses and nonnegative, nonzero intensities",
            )

        row = metadata.iloc[row_index]

        try:
            precursor = float(row.get("precursor_mz"))
        except (ValueError, TypeError):
            precursor = np.nan

        if error is None and (not np.isfinite(precursor) or precursor <= 0):
            error = RejectedRecord(
                "invalid_precursor",
                "Missing, nonnumeric, or nonpositive precursor_mz",
            )

        mode = text_value(row.get("ionmode")).lower()
        if error is None:
            try:
                mode = validate_ionmode(mode, library_ionmode)
            except RejectedRecord as exc:
                error = exc

        if error is None:
            precursors[row_index] = precursor

        errors.append(error)
        ionmodes.append(mode)

    return errors, precursors, tuple(ionmodes)


def _process_collection_resilient(
    processor: SpectraProcessor,
    collection: SpectraCollection,
) -> tuple[list[SpectraCollection], dict[int, RejectedRecord]]:
    """Run collection processing while isolating exceptional rows.

    Normal matchms requirement filters remove rows without raising. If a filter
    raises because of malformed input, recursive bisection identifies the
    failing row instead of rejecting the complete query batch.
    """
    try:
        processed = processor.process_collection(collection)
    except (ValueError, TypeError, AssertionError) as exc:
        if len(collection) == 1:
            row_id = int(collection.metadata.iloc[0][_QUERY_ROW_ID])
            return [], {
                row_id: RejectedRecord("query_processing", str(exc))
            }

        middle = len(collection) // 2
        left, left_errors = _process_collection_resilient(
            processor,
            collection[:middle],
        )
        right, right_errors = _process_collection_resilient(
            processor,
            collection[middle:],
        )
        return left + right, {**left_errors, **right_errors}

    if processed is None or len(processed) == 0:
        return [], {}

    return [processed], {}


def process_query_collection(
    spectra: Sequence[Spectrum],
    spec: ProcessingSpec,
    *,
    library_ionmode: str,
    mz_precision: float,
    processor: SpectraProcessor | None = None,
) -> ProcessedQueryCollection:
    """Preprocess and validate a batch of query spectra as one collection.

    A temporary row identifier is attached before matchms processing so that
    collection-level filters may freely remove rows without losing alignment to
    the caller's input sequence. Structural annotations are removed before any
    model inference or spectral scoring.
    """
    from matchms import SpectraCollection

    spectra = list(spectra)
    if not spectra:
        return ProcessedQueryCollection(
            spectra=None,
            input_rows=(),
            precursor_mz=np.empty(0, dtype=np.float64),
            ionmodes=(),
            rejected={},
        )

    processor = processor or make_processor(spec)
    if not callable(getattr(processor, "process_collection", None)):
        raise TypeError("processor must expose process_collection()")

    collection = SpectraCollection(
        spectra,
        mz_precision=mz_precision,
    )
    collection.add_metadata(
        list(range(len(collection))),
        col_name=_QUERY_ROW_ID,
    )

    # The collection is newly owned here, but use the public non-mutating helper
    # to keep the no-caller-mutation contract obvious.
    clean = strip_structure_annotations(collection)

    processed_parts, rejected = _process_collection_resilient(
        processor,
        clean,
    )

    if not processed_parts:
        # Rows that disappeared normally were filtered by a requirement filter;
        # rows that raised already have a more specific processing error.
        for input_row in range(len(spectra)):
            rejected.setdefault(
                input_row,
                RejectedRecord(
                    "filtered_spectrum",
                    "Spectrum did not satisfy query processing requirements",
                ),
            )
        return ProcessedQueryCollection(
            spectra=None,
            input_rows=(),
            precursor_mz=np.empty(0, dtype=np.float64),
            ionmodes=(),
            rejected=rejected,
        )

    if len(processed_parts) == 1:
        processed = processed_parts[0]
    else:
        processed = SpectraCollection.concat(processed_parts)

    retained_rows = tuple(
        int(value)
        for value in processed.metadata[_QUERY_ROW_ID].to_numpy()
    )
    retained_set = set(retained_rows)

    for input_row in range(len(spectra)):
        if input_row not in retained_set:
            rejected.setdefault(
                input_row,
                RejectedRecord(
                    "filtered_spectrum",
                    "Spectrum did not satisfy query processing requirements",
                ),
            )

    errors, precursors, ionmodes = validate_collection(
        processed,
        library_ionmode,
    )

    keep = np.ones(len(processed), dtype=bool)
    for row_index, error in enumerate(errors):
        if error is None:
            continue
        input_row = retained_rows[row_index]
        rejected[input_row] = error
        keep[row_index] = False

    if not np.any(keep):
        return ProcessedQueryCollection(
            spectra=None,
            input_rows=(),
            precursor_mz=np.empty(0, dtype=np.float64),
            ionmodes=(),
            rejected=rejected,
        )

    valid = processed.filter(keep)
    valid_input_rows = tuple(
        retained_rows[index]
        for index in np.flatnonzero(keep)
    )
    valid_precursors = precursors[keep]
    valid_ionmodes = tuple(
        ionmodes[index]
        for index in np.flatnonzero(keep)
    )

    valid = valid.drop_metadata(
        _QUERY_ROW_ID,
        errors="ignore",
    )

    return ProcessedQueryCollection(
        spectra=valid,
        input_rows=valid_input_rows,
        precursor_mz=valid_precursors,
        ionmodes=valid_ionmodes,
        rejected=rejected,
    )


def spectrum_from_record(row) -> Spectrum:
    """Reconstruct one canonical stored reference spectrum.

    Stored peaks are already cleaned and are therefore reconstructed verbatim;
    no matchms processing pipeline is applied again.
    """
    from matchms import Spectrum

    metadata = dict(row.get("other_metadata") or {})
    metadata.update(
        precursor_mz=row["precursor_mz"],
        ionmode=row["ionmode"],
    )

    mz = np.frombuffer(
        bytes(row["mz_values"]),
        dtype="<f8",
    ).copy()
    intensity = np.frombuffer(
        bytes(row["intensity_values"]),
        dtype="<f8",
    ).copy()

    if len(mz) != row["n_peaks"] or len(intensity) != row["n_peaks"]:
        raise ContractError(
            "Stored peak payload length does not match n_peaks"
        )

    return Spectrum(
        mz=mz,
        intensities=intensity,
        metadata=metadata,
        metadata_harmonization=False,
    )


def collection_from_records(
    rows,
    *,
    mz_precision: float,
) -> SpectraCollection | None:
    """Reconstruct stored reference rows as a ``SpectraCollection``.

    ``None`` is returned for an empty input because matchms intentionally does
    not construct empty ``SpectraCollection`` objects.
    """
    from matchms import SpectraCollection

    rows = list(rows)
    if not rows:
        return None

    return SpectraCollection(
        [spectrum_from_record(row) for row in rows],
        mz_precision=mz_precision,
    )

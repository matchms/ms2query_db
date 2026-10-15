"""Thin adapter around the matchms indexed-search API.

This module is supposed to be the only part of MS2Query-DB that knows details of
matchms native similarity classes, ``Scores`` objects, and index persistence.
Keeping that boundary small makes the rest of the project independent of minor
matchms API details and gives index manifests one stable implementation identity.
(that's the idea at least)
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from ..errors import ContractError
from ..util import file_sha256, scientific_versions

_NATIVE_METHODS = {
    "cosine",
    "modified_cosine",
    "entropy_fragment",
    "entropy_hybrid",
}


def make_scorer(method, spec):
    """Construct the configured matchms scorer for one classical method.

    Parameters
    ----------
    method
        One of the classical search method names supported by MS2Query-DB.
    spec
        :class:`~ms2query_db.config.ScoringSpec` containing the shared native
        scoring parameters.

    Returns
    -------
    matchms.similarity.BaseSimilarity
        A scorer exposing matchms' indexed ``build_index``/``search`` API.

    Notes
    -----
    ``entropy_fragment`` uses :class:`matchms.similarity.EntropySearch`, the
    fragment-only entropy implementation optimized for indexed library search.
    ``entropy_hybrid`` uses :class:`matchms.similarity.Entropy` in hybrid mode.
    """
    if method not in _NATIVE_METHODS:
        raise ValueError(
            f"Not a native classical method: {method!r}. "
            f"Choose from {sorted(_NATIVE_METHODS)}."
        )

    from matchms.similarity import Cosine, Entropy, EntropySearch, ModifiedCosine

    common = {
        "tolerance": spec.tolerance,
        "noise_cutoff": spec.noise_cutoff,
        "remove_precursor": spec.remove_precursor,
        "offset_to_precursor": spec.offset_to_precursor,
        "dtype": np.dtype(spec.dtype),
    }

    if method == "cosine":
        return Cosine(
            intensity_power=spec.intensity_power,
            **common,
        )
    if method == "modified_cosine":
        return ModifiedCosine(
            intensity_power=spec.intensity_power,
            **common,
        )
    if method == "entropy_fragment":
        return EntropySearch(**common)

    return Entropy(
        matching_mode="hybrid",
        **common,
    )


def scorer_identity(scorer) -> dict:
    """Return a reproducibility identity for a native scorer implementation.

    Development versions of matchms can change without changing their version
    string. For that reason the identity includes SHA256 digests of Python
    sources in the scorer's package in addition to the reported matchms version.
    Native index manifests therefore become invalid when the implementation that
    created them changes.
    """
    scorer_type = type(scorer)
    module = inspect.getmodule(scorer_type)
    source_file = (
        Path(module.__file__).resolve()
        if module is not None and getattr(module, "__file__", None)
        else None
    )

    sources: dict[str, str] = {}
    if source_file is not None:
        root = source_file.parent
        for path in sorted(root.rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            sources[str(path.relative_to(root))] = file_sha256(path)

    versions = scientific_versions()
    return {
        "class": scorer_type.__name__,
        "module": scorer_type.__module__,
        "matchms": versions.get("matchms"),
        "sources": sources,
    }


def require_index_api(scorer) -> None:
    """Require the matchms development indexed-search API on ``scorer``."""
    required = (
        "build_index",
        "search",
        "save_index",
        "load_index",
    )
    missing = [
        name
        for name in required
        if not callable(getattr(scorer, name, None))
    ]
    if missing:
        raise ContractError(
            f"{type(scorer).__name__} does not expose the required matchms "
            f"indexed-search API; missing {missing}."
        )


def build_index(scorer, spectra):
    """Build one native index from a collection after checking the API contract."""
    require_index_api(scorer)
    return scorer.build_index(spectra)


def search_arrays(
    scorer,
    queries,
    index,
    *,
    expected_columns: int,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Search a native index and return validated dense score arrays.

    Rows always correspond to query spectra and columns to indexed reference
    spectra. Match counts are returned when the scorer exposes a ``matches``
    field; entropy implementations, for example, may return scores only.
    """
    require_index_api(scorer)

    if (
        isinstance(expected_columns, bool)
        or not isinstance(expected_columns, int)
        or expected_columns < 0
    ):
        raise ValueError("expected_columns must be a nonnegative integer.")
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError("n_jobs must be a positive integer or -1.")

    parameters = inspect.signature(scorer.search).parameters
    optional = {
        "progress_bar": False,
        "n_jobs": n_jobs,
    }
    kwargs = {
        key: value
        for key, value in optional.items()
        if key in parameters
    }

    result = scorer.search(
        queries,
        index,
        **kwargs,
    )
    if not callable(getattr(result, "to_array", None)):
        raise ContractError(
            "matchms scorer.search() must return a Scores-like object exposing "
            "to_array()."
        )

    scores = np.asarray(
        result.to_array("score"),
        dtype=np.float64,
    )
    expected = (len(queries), expected_columns)
    if scores.shape != expected:
        raise ContractError(
            f"Native score array has shape {scores.shape}; expected query x "
            f"library shape {expected}."
        )
    if not np.isfinite(scores).all():
        raise ContractError(
            "Native scorer produced nonfinite scores."
        )

    try:
        matches = np.asarray(result.to_array("matches"))
    except (KeyError, ValueError):
        matches = None

    if matches is not None:
        if matches.shape != expected:
            raise ContractError(
                "Native match-count shape differs from the score shape."
            )
        if not np.issubdtype(matches.dtype, np.number):
            raise ContractError("Native match counts must be numeric.")
        if not np.isfinite(matches).all():
            raise ContractError("Native scorer produced nonfinite match counts.")

    return scores, matches


def save_index(scorer, index, path: str | Path) -> None:
    """Persist a native index and verify that the requested file was written."""
    require_index_api(scorer)
    path = Path(path)
    parameters = inspect.signature(scorer.save_index).parameters
    kwargs = {"overwrite": True} if "overwrite" in parameters else {}
    scorer.save_index(index, path, **kwargs)

    if not path.is_file():
        raise ContractError(
            f"Native save_index() did not write the requested path: {path}"
        )


def load_index(scorer, path: str | Path):
    """Load a persisted native index through the checked matchms adapter."""
    require_index_api(scorer)
    path = Path(path)
    if not path.is_file():
        raise ContractError(f"Native index file does not exist: {path}")
    return scorer.load_index(path)


def index_spectrum_count(index) -> int | None:
    """Return an index's reference-spectrum count when the backend exposes it."""
    value = getattr(index, "n_specs", None)
    if value is None:
        return None
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ContractError("Native index exposes an invalid n_specs value.") from exc
    if count < 0:
        raise ContractError("Native index exposes a negative n_specs value.")
    return count


def top_indices(scores, ids, n: int) -> np.ndarray:
    """Return exact top-N column indices with deterministic ID tie breaking.

    A complete argsort is avoided for large shards. Candidates above the cutoff
    are retained directly and ties at the cutoff are resolved by increasing
    spectrum ID. The returned indices themselves are ordered by descending score
    and then ascending spectrum ID.
    """
    values = np.asarray(scores)
    spectrum_ids = np.asarray(ids)

    if values.ndim != 1 or spectrum_ids.ndim != 1 or values.shape != spectrum_ids.shape:
        raise ContractError(
            "Native top-N scores and spectrum IDs must be aligned one-dimensional arrays."
        )
    if not np.isfinite(values).all():
        raise ContractError("Native top-N scores must be finite.")
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer.")

    n = min(n, len(values))
    if n == 0:
        return np.empty(0, dtype=np.int64)

    if n == len(values):
        selected = np.arange(n, dtype=np.int64)
    else:
        cutoff_position = len(values) - n
        cutoff = np.partition(values, cutoff_position)[cutoff_position]
        above = np.flatnonzero(values > cutoff)
        ties = np.flatnonzero(values == cutoff)
        take = n - len(above)

        if take < len(ties):
            tie_ids = spectrum_ids[ties]
            tie_order = np.argpartition(tie_ids, take - 1)[:take]
            ties = ties[tie_order]
        else:
            ties = ties[:take]

        selected = np.concatenate((above, ties))

    order = np.lexsort(
        (spectrum_ids[selected], -values[selected])
    )
    return selected[order].astype(np.int64, copy=False)

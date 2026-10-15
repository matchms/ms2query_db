"""MS2DeepScore model loading and robust batch embedding inference.

This module has one narrow responsibility: turn processed spectra into numeric
embedding vectors.  It does not search the database and it does not decide
which reference spectra are nearest neighbours.

Two layers are kept separate:

``MS2DeepScoreEmbedder``
    A small adapter around the MS2DeepScore API.  It loads a model, exposes a
    stable ``embed()`` method, and records the vector dimension and model ID.

``validated_embeddings``
    Runs an embedder on a batch while preserving query/reference row positions.
    A bad individual spectrum is returned as a :class:`RejectedRecord` instead
    of discarding the complete batch.  Successful vectors are checked and L2
    normalized before they are stored or searched.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Protocol

import numpy as np

from .errors import ContractError, RejectedRecord
from .util import file_sha256

DEFAULT_EMBEDDING_BATCH_SIZE = 256
MAX_STORED_EMBEDDING_DIMENSION = 2000


class SpectrumEmbedder(Protocol):
    """Minimal interface required by the rest of MS2Query-DB.

    Custom embedders can implement this protocol as long as they expose a
    stable vector ``dimension``, a descriptive ``model_id``, and an ``embed``
    method that returns one vector for every input spectrum.
    """

    dimension: int
    model_id: str

    def embed(self, spectra) -> np.ndarray:
        """Return one embedding vector per input spectrum."""
        ...


def _row_count(value) -> int:
    """Return the number of rows in an embedder result or raise a clear error."""
    try:
        return len(value)
    except TypeError as exc:
        raise ContractError(
            "Embedder returned an object without a row dimension."
        ) from exc


def _to_cpu_array(value):
    """Convert common tensor outputs to a NumPy-compatible object.

    MS2DeepScore can use PyTorch internally.  Keeping this small conversion here
    avoids importing torch as a hard dependency of this module.
    """
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return value


def _validate_embedding_row(
    row,
    *,
    dimension: int,
) -> np.ndarray | RejectedRecord:
    """Validate and L2-normalize one embedding vector.

    Problems caused by the individual spectrum (missing vector, non-numeric
    values, NaN/inf, or a zero vector) become ``RejectedRecord`` objects.  A
    wrong vector *dimension* is different: that indicates an incompatible or
    broken embedder and therefore raises :class:`ContractError`.
    """
    if row is None:
        return RejectedRecord(
            "empty_embedding",
            "Embedding is missing.",
        )

    try:
        vector = np.asarray(row, dtype=np.float64)
    except (TypeError, ValueError):
        return RejectedRecord(
            "invalid_embedding",
            "Embedding is not numeric.",
        )

    if vector.size == 0:
        return RejectedRecord(
            "empty_embedding",
            "Embedding has no values.",
        )

    expected_shape = (dimension,)
    if vector.shape != expected_shape:
        raise ContractError(
            f"Embedding shape {vector.shape} does not match the model "
            f"dimension {expected_shape}."
        )

    if not np.isfinite(vector).all():
        return RejectedRecord(
            "invalid_embedding",
            "Embedding contains NaN or infinity.",
        )

    # Scaling by the largest absolute component before computing the L2 norm
    # avoids overflow/underflow for unusually large or small model outputs.
    scale = float(np.max(np.abs(vector)))
    if scale == 0.0:
        return RejectedRecord(
            "empty_embedding",
            "Embedding has zero norm.",
        )

    scaled = vector / scale
    norm = float(np.linalg.norm(scaled))
    if not np.isfinite(norm) or norm == 0.0:
        return RejectedRecord(
            "invalid_embedding",
            "Embedding could not be normalized to a finite vector.",
        )

    return (scaled / norm).astype(np.float32)


def _validate_embedding_batch(
    raw_embeddings,
    *,
    expected_rows: int,
    dimension: int,
) -> list[np.ndarray | RejectedRecord]:
    """Validate one row-aligned batch returned by an embedder."""
    raw_embeddings = _to_cpu_array(raw_embeddings)
    actual_rows = _row_count(raw_embeddings)

    if actual_rows != expected_rows:
        raise ContractError(
            "Embedder changed row count: "
            f"returned {actual_rows} row(s) for {expected_rows} spectrum/spectra."
        )

    return [
        _validate_embedding_row(row, dimension=dimension)
        for row in raw_embeddings
    ]


def validated_embeddings(
    embedder: SpectrumEmbedder,
    spectra,
) -> list[np.ndarray | RejectedRecord]:
    """Embed spectra while keeping one output entry per input row.

    Parameters
    ----------
    embedder
        Object implementing :class:`SpectrumEmbedder`.
    spectra
        A row-addressable spectrum collection, normally a matchms
        ``SpectraCollection``.

    Returns
    -------
    list[numpy.ndarray | RejectedRecord]
        One item for every input spectrum, in the same order.  Valid embeddings
        are finite, L2-normalized ``float32`` vectors.  Problems attributable
        to one spectrum are represented by ``RejectedRecord`` in that row.

    Notes
    -----
    Some embedding implementations raise ``ValueError`` for an entire batch
    when one spectrum is unsuitable.  To preserve source/query alignment, this
    function retries a failing batch by splitting it in half until the bad row
    is isolated.  Healthy batches therefore require only one model call; the
    extra calls occur only after an embedding failure.

    A changed row count or vector dimension is treated as a programming/model
    contract error and raises :class:`ContractError` immediately.
    """
    n_spectra = len(spectra)
    if n_spectra == 0:
        return []

    try:
        raw_embeddings = embedder.embed(spectra)
    except ValueError as exc:
        if n_spectra == 1:
            return [
                RejectedRecord(
                    "embedding_error",
                    str(exc),
                )
            ]

        middle = n_spectra // 2
        return validated_embeddings(
            embedder,
            spectra[:middle],
        ) + validated_embeddings(
            embedder,
            spectra[middle:],
        )

    return _validate_embedding_batch(
        raw_embeddings,
        expected_rows=n_spectra,
        dimension=embedder.dimension,
    )


def _embedding_method(scorer):
    """Return the supported MS2DeepScore vector-inference method."""
    for name in (
        "get_embedding_array",
        "calculate_vectors",
    ):
        method = getattr(scorer, name, None)
        if callable(method):
            return method

    raise ContractError(
        "MS2DeepScore does not expose get_embedding_array() or "
        "calculate_vectors()."
    )


def _supported_keyword_arguments(callable_, values: dict) -> dict:
    """Keep only optional arguments accepted by an external callable."""
    try:
        parameters = inspect.signature(callable_).parameters
    except (TypeError, ValueError):
        return {}

    return {
        name: value
        for name, value in values.items()
        if name in parameters
    }


class MS2DeepScoreEmbedder:
    """Stable adapter around an MS2DeepScore scorer.

    MS2DeepScore has used more than one public method name for extracting
    vectors across releases.  This adapter hides that compatibility detail from
    the builder and query engine and exposes one method: :meth:`embed`.

    Parameters
    ----------
    scorer
        Configured ``ms2deepscore.MS2DeepScore`` instance.
    model_id
        Stable identifier for the model.  ``from_file()`` uses the SHA-256 of
        the model file, which allows a built library to record exactly which
        model produced its stored reference embeddings.
    batch_size
        Internal inference batch size passed to MS2DeepScore when that API
        supports it.

    Notes
    -----
    The current PostgreSQL/pgvector HNSW storage used by MS2Query-DB supports
    at most 2000 vector dimensions, so model dimensions are checked here before
    a long library build begins.
    """

    def __init__(
        self,
        scorer,
        model_id: str,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a non-empty string.")
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("batch_size must be a positive integer.")

        try:
            dimension = int(getattr(scorer, "output_vector_dim"))
        except (TypeError, ValueError, AttributeError) as exc:
            raise ContractError(
                "MS2DeepScore scorer does not expose a valid output_vector_dim."
            ) from exc

        if not 1 <= dimension <= MAX_STORED_EMBEDDING_DIMENSION:
            raise ContractError(
                "MS2Query-DB requires an embedding dimension between 1 and "
                f"{MAX_STORED_EMBEDDING_DIMENSION}; got {dimension}."
            )

        self.scorer = scorer
        self.model_id = model_id
        self.batch_size = batch_size
        self.dimension = dimension
        self._embed_method = _embedding_method(scorer)
        self._embed_kwargs = _supported_keyword_arguments(
            self._embed_method,
            {
                "batch_size": batch_size,
                "progress_bar": False,
                "datatype": "numpy",
            },
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        *,
        allow_legacy: bool = False,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> "MS2DeepScoreEmbedder":
        """Load an MS2DeepScore model file and create an embedder.

        ``allow_legacy=False`` is intentionally the default.  If the installed
        MS2DeepScore version cannot express a safe/legacy loading choice, the
        method refuses to load implicitly and asks the caller to make the trust
        decision explicit.
        """
        if not isinstance(allow_legacy, bool):
            raise ValueError("allow_legacy must be a boolean.")

        from ms2deepscore import MS2DeepScore
        from ms2deepscore.models import load_model

        model_path = Path(path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(model_path)

        load_parameters = _supported_keyword_arguments(
            load_model,
            {"allow_legacy": allow_legacy},
        )

        try:
            loader_signature = inspect.signature(load_model).parameters
        except (TypeError, ValueError):
            loader_signature = {}

        if "allow_legacy" not in loader_signature and not allow_legacy:
            raise RuntimeError(
                "The installed MS2DeepScore loader has no safe-first "
                "allow_legacy option. Use a trusted model with "
                "allow_legacy=True, or upgrade MS2DeepScore."
            )

        model = load_model(
            model_path,
            **load_parameters,
        )

        scorer_kwargs = _supported_keyword_arguments(
            MS2DeepScore,
            {"progress_bar": False},
        )
        scorer = MS2DeepScore(
            model,
            **scorer_kwargs,
        )

        return cls(
            scorer,
            model_id=f"sha256:{file_sha256(model_path)}",
            batch_size=batch_size,
        )

    def embed(self, spectra):
        """Return raw MS2DeepScore vectors for a spectrum collection.

        Validation and L2 normalization intentionally happen in
        :func:`validated_embeddings`, so custom embedders and MS2DeepScore use
        exactly the same downstream checks.
        """
        return self._embed_method(
            spectra,
            **self._embed_kwargs,
        )

"""Chemical structure handling, fingerprints, projections, and similarities.

This module contains the chemistry-specific pieces used by MS2Query-DB:

1. validate and canonicalize molecular structures;
2. calculate library fingerprints with ``chemap``/RDKit;
3. serialize fingerprints for PostgreSQL storage;
4. project high-dimensional fingerprints into vectors used only for ANN
   candidate retrieval;
5. calculate exact chemical similarities in the original fingerprint space.

The projected vectors are deliberately *not* used as the final chemical
similarity measure. They only provide an efficient shortlist for database
search; exact comparisons are performed with the stored fingerprints.
"""

from __future__ import annotations

import hashlib
import inspect
import zlib
from dataclasses import asdict, dataclass
from functools import lru_cache

import numpy as np
from scipy.sparse import csr_matrix

from .config import FingerprintSpec
from .errors import ContractError, RejectedRecord
from .util import text_value

FLOAT32_EXACT_INTEGER_LIMIT = 2**24
"""Largest integer that can be represented exactly by every float32 value."""


@dataclass(frozen=True)
class CompoundIdentity:
    """Canonical structural identity stored for one compound.

    ``identity_key`` is the database-level unique structure key. It is the
    SHA-256 digest of canonical isomeric SMILES, so stereochemically distinct
    canonical SMILES remain distinct compounds.

    ``inchikey14`` is intentionally *not* unique. It is useful for grouping
    related structures, for example when constructing train/test splits, but
    several valid structures may share the same first InChIKey block.
    """

    identity_key: str
    inchikey14: str
    inchikey: str
    smiles: str
    inchi: str
    formula: str
    exact_mass: float

    def to_dict(self) -> dict[str, object]:
        """Return the identity as a plain dictionary suitable for DB rows."""
        return asdict(self)


def _clean_structure_inputs(
    smiles: str,
    inchi: str,
    inchikey: str,
) -> tuple[str, str, str]:
    """Normalize textual identifiers and reject unsafe embedded NUL bytes."""
    values = tuple(text_value(value) for value in (smiles, inchi, inchikey))

    if any("\x00" in value for value in values):
        raise RejectedRecord(
            "invalid_structure",
            "Structural identifier contains a NUL character",
        )

    return values


def _molecule_from_structure(smiles: str, inchi: str):
    """Construct an RDKit molecule from SMILES, falling back to InChI."""
    from rdkit import Chem

    if not smiles and not inchi:
        raise RejectedRecord(
            "missing_structure",
            "No SMILES or InChI supplied",
        )

    try:
        molecule = (
            Chem.MolFromSmiles(smiles)
            if smiles
            else Chem.MolFromInchi(inchi)
        )
    except (ValueError, RuntimeError) as exc:
        raise RejectedRecord("invalid_structure", str(exc)) from exc

    if molecule is None or molecule.GetNumAtoms() == 0:
        raise RejectedRecord(
            "invalid_structure",
            "SMILES/InChI is unusable or has no atoms",
        )

    return molecule


def _canonical_identifiers(molecule) -> tuple[str, str, str, str, float]:
    """Generate canonical identifiers and basic molecular properties."""
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    try:
        inchikey = Chem.MolToInchiKey(molecule)
        smiles = Chem.MolToSmiles(
            molecule,
            canonical=True,
            isomericSmiles=True,
        )
        inchi = Chem.MolToInchi(molecule)
        formula = rdMolDescriptors.CalcMolFormula(molecule)
        exact_mass = float(rdMolDescriptors.CalcExactMolWt(molecule))
    except (ValueError, RuntimeError) as exc:
        raise RejectedRecord("structure_conversion", str(exc)) from exc

    if (
        len(inchikey) != 27
        or not inchi
        or not smiles
        or not np.isfinite(exact_mass)
    ):
        raise RejectedRecord(
            "structure_conversion",
            "Cannot generate consistent structural identifiers",
        )

    return inchikey, smiles, inchi, formula, exact_mass


def _validate_supplied_inchikey(
    supplied: str,
    generated: str,
) -> None:
    """Check an optional supplied 14- or 27-character InChIKey."""
    if not supplied:
        return

    if len(supplied) not in {14, 27}:
        raise RejectedRecord(
            "identifier_mismatch",
            "Provided InChIKey must contain 14 or 27 characters",
        )

    if generated[: len(supplied)] != supplied:
        raise RejectedRecord(
            "identifier_mismatch",
            "Provided InChIKey disagrees with the structure",
        )


def _validate_smiles_inchi_agreement(
    *,
    supplied_smiles: str,
    supplied_inchi: str,
    generated_inchikey: str,
) -> None:
    """Verify that simultaneously supplied SMILES and InChI describe one molecule."""
    if not supplied_smiles or not supplied_inchi:
        return

    from rdkit import Chem

    try:
        inchi_molecule = Chem.MolFromInchi(supplied_inchi)
        inchi_key = (
            Chem.MolToInchiKey(inchi_molecule)
            if inchi_molecule is not None
            else None
        )
    except (ValueError, RuntimeError) as exc:
        raise RejectedRecord(
            "identifier_mismatch",
            "Provided InChI cannot be validated",
        ) from exc

    if inchi_key != generated_inchikey:
        raise RejectedRecord(
            "identifier_mismatch",
            "Provided SMILES and InChI disagree",
        )


@lru_cache(maxsize=20_000)
def identify_compound(
    smiles: str,
    inchi: str = "",
    inchikey: str = "",
) -> CompoundIdentity:
    """Validate a structure and return the canonical compound identity.

    The function intentionally does not repair salts, neutralize structures,
    standardize tautomers, or replace one submitted identifier with another.
    Such operations can change chemical identity and should be explicit
    preprocessing decisions outside the database layer.

    Canonical isomeric SMILES defines the unique structure stored by
    MS2Query-DB. Its SHA-256 digest becomes ``identity_key``. The complete
    InChIKey and first 14 characters are stored as useful identifiers/grouping
    labels, but ``inchikey14`` is not treated as unique.

    If both SMILES and InChI are supplied, they must resolve to the same
    InChIKey. If an InChIKey is supplied, its 14- or 27-character value must
    agree with the generated structure.

    Raises
    ------
    RejectedRecord
        If the identifiers are missing, unparsable, mutually inconsistent, or
        cannot be converted into a complete canonical representation.
    """
    smiles, inchi, inchikey = _clean_structure_inputs(
        smiles,
        inchi,
        inchikey,
    )

    molecule = _molecule_from_structure(smiles, inchi)
    (
        generated_key,
        canonical_smiles,
        canonical_inchi,
        formula,
        exact_mass,
    ) = _canonical_identifiers(molecule)

    _validate_supplied_inchikey(inchikey, generated_key)
    _validate_smiles_inchi_agreement(
        supplied_smiles=smiles,
        supplied_inchi=inchi,
        generated_inchikey=generated_key,
    )

    identity_key = hashlib.sha256(
        canonical_smiles.encode("utf-8")
    ).hexdigest()

    return CompoundIdentity(
        identity_key=identity_key,
        inchikey14=generated_key[:14],
        inchikey=generated_key,
        smiles=canonical_smiles,
        inchi=canonical_inchi,
        formula=formula,
        exact_mass=exact_mass,
    )


class ChemapFingerprinter:
    """Calculate one validated fingerprint result for every input SMILES.

    ``chemap`` performs the actual batched calculation, while this adapter
    provides the row-alignment and validation guarantees needed by the import
    pipeline.

    A scientifically invalid input row becomes ``RejectedRecord`` without
    shifting later rows. If chemap raises ``ValueError`` for a batch, the batch
    is recursively split until the failing individual structure is isolated.
    Programming errors, memory errors, native crashes, and other unexpected
    exceptions are deliberately *not* swallowed.
    """

    def __init__(self, spec: FingerprintSpec):
        from chemap import FingerprintConfig, compute_fingerprints
        from rdkit.Chem import rdFingerprintGenerator

        self.spec = spec
        self.generator = self._make_generator(rdFingerprintGenerator)
        self._compute = compute_fingerprints
        self._config = FingerprintConfig(
            count=spec.count,
            folded=True,
            return_csr=False,
            invalid_policy="raise",
        )
        self._compute_kwargs = self._supported_compute_kwargs()

    def _make_generator(self, generators):
        """Create the RDKit fingerprint generator described by the config."""
        if self.spec.family == "morgan":
            return generators.GetMorganGenerator(
                radius=self.spec.radius,
                fpSize=self.spec.size,
                includeChirality=self.spec.include_chirality,
            )

        return generators.GetRDKitFPGenerator(
            fpSize=self.spec.size,
        )

    def _supported_compute_kwargs(self) -> dict[str, object]:
        """Build kwargs supported by the installed chemap version."""
        parameters = inspect.signature(self._compute).parameters
        kwargs: dict[str, object] = {"config": self._config}

        if "n_jobs" in parameters:
            kwargs["n_jobs"] = self.spec.n_jobs

        if "show_progress" in parameters:
            kwargs["show_progress"] = False

        return kwargs

    def compute_many(
        self,
        smiles: list[str],
    ) -> list[np.ndarray | RejectedRecord]:
        """Return one fingerprint or rejection for every input SMILES.

        Successful fingerprints are returned as float32 arrays of length
        ``FingerprintSpec.size``. Output order always matches input order.
        """
        if not smiles:
            return []

        try:
            raw = self._compute(
                smiles,
                self.generator,
                **self._compute_kwargs,
            )
        except ValueError as exc:
            return self._isolate_failed_batch(smiles, exc)

        matrix = np.asarray(raw)
        expected_shape = (len(smiles), self.spec.size)

        if matrix.shape != expected_shape:
            raise ContractError(
                f"Chemap returned {matrix.shape}; expected {expected_shape}"
            )

        return [
            self._validate_row(row)
            for row in matrix
        ]

    def _isolate_failed_batch(
        self,
        smiles: list[str],
        error: ValueError,
    ) -> list[np.ndarray | RejectedRecord]:
        """Recursively isolate structures that make batched chemap fail."""
        if len(smiles) == 1:
            return [
                RejectedRecord(
                    "fingerprint_error",
                    str(error),
                )
            ]

        middle = len(smiles) // 2
        return (
            self.compute_many(smiles[:middle])
            + self.compute_many(smiles[middle:])
        )

    def _validate_row(
        self,
        row,
    ) -> np.ndarray | RejectedRecord:
        """Validate one raw fingerprint and apply its configured transform."""
        values = np.asarray(row, dtype=np.float64)

        if not np.isfinite(values).all() or np.any(values < 0):
            return RejectedRecord(
                "invalid_fingerprint",
                "Fingerprint contains nonfinite or negative values",
            )

        if not np.any(values > 0):
            return RejectedRecord(
                "empty_fingerprint",
                "Selected fingerprint representation yields no features",
            )

        if (
            np.any(values != np.floor(values))
            or np.any(values > FLOAT32_EXACT_INTEGER_LIMIT)
        ):
            return RejectedRecord(
                "invalid_fingerprint",
                "Raw fingerprint counts must be nonnegative integers that "
                "remain exact in float32",
            )

        if not self.spec.count and np.any(values > 1):
            raise ContractError(
                "Chemap returned count values for a binary fingerprint request"
            )

        if self.spec.value_transform == "log1p":
            values = np.log1p(values)

        return values.astype(np.float32)


def _validate_fingerprint_vector(
    values,
    *,
    error_type: type[Exception],
    message: str,
) -> np.ndarray:
    """Return a validated one-dimensional nonnegative, nonempty vector."""
    vector = np.asarray(values, dtype=np.float32)

    if (
        vector.ndim != 1
        or not np.isfinite(vector).all()
        or np.any(vector < 0)
        or not np.any(vector > 0)
    ):
        raise error_type(message)

    return vector


def encode_fingerprint(values) -> bytes:
    """Serialize one validated fingerprint into a compact compressed blob.

    Fingerprints are stored as little-endian float32 values and compressed
    with zlib. Validation happens before serialization so invalid vectors are
    never written to the database.
    """
    vector = _validate_fingerprint_vector(
        values,
        error_type=ContractError,
        message="Only valid, nonempty fingerprints may be stored",
    ).astype("<f4", copy=False)

    return zlib.compress(
        vector.tobytes(),
        level=3,
    )


def decode_fingerprint(payload, size: int) -> np.ndarray:
    """Decode and validate one fingerprint stored by :func:`encode_fingerprint`.

    Decompression is deliberately bounded to the expected byte count. This
    avoids trusting a malformed/corrupt compressed payload to allocate an
    arbitrarily large output buffer.
    """
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")

    expected_bytes = size * np.dtype("<f4").itemsize
    stream = zlib.decompressobj()
    raw = stream.decompress(
        bytes(payload),
        expected_bytes + 1,
    )

    if (
        len(raw) != expected_bytes
        or not stream.eof
        or stream.unused_data
    ):
        raise ContractError(
            "Corrupt fingerprint payload or wrong fingerprint dimension"
        )

    vector = np.frombuffer(
        raw,
        dtype="<f4",
    ).copy()

    return _validate_fingerprint_vector(
        vector,
        error_type=ContractError,
        message="Stored fingerprint is invalid",
    )


class FingerprintProjector:
    """Project fingerprints into compact unit vectors for ANN retrieval.

    PostgreSQL/pgvector searches a lower-dimensional vector efficiently. The
    projection therefore provides a *candidate shortlist only*. Final chemical
    similarities are always recalculated in the original fingerprint space.

    The projection matrix is deterministic for a given ``FingerprintSpec`` so
    a library can store the matrix once and reproduce it exactly when reopened.
    """

    def __init__(self, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)

        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ContractError("Invalid projection matrix")

        self.matrix = matrix

    @classmethod
    def create(cls, spec: FingerprintSpec) -> "FingerprintProjector":
        """Create the deterministic Gaussian projection for a fingerprint spec."""
        rng = np.random.Generator(
            np.random.PCG64(spec.projection_seed)
        )

        matrix = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(spec.projection_dim),
            size=(spec.size, spec.projection_dim),
        )

        return cls(matrix)

    def transform(self, vectors) -> np.ndarray:
        """Project fingerprint rows and L2-normalize the resulting vectors."""
        matrix = np.asarray(vectors, dtype=np.float32)

        if (
            matrix.ndim != 2
            or matrix.shape[1] != self.matrix.shape[0]
            or not np.isfinite(matrix).all()
            or np.any(matrix < 0)
        ):
            raise ContractError(
                "Projection input has invalid dimensions or values"
            )

        projected = np.asarray(
            csr_matrix(matrix) @ self.matrix,
            dtype=np.float64,
        )

        norms = np.linalg.norm(projected, axis=1)

        if not np.isfinite(norms).all() or np.any(norms == 0):
            raise ContractError(
                "Nonzero fingerprints produced invalid or zero projected vectors"
            )

        return (
            projected / norms[:, None]
        ).astype(np.float32)

    def to_bytes(self) -> bytes:
        """Serialize the projection matrix as little-endian float32 bytes."""
        return self.matrix.astype(
            "<f4",
            copy=False,
        ).tobytes()

    @classmethod
    def from_bytes(
        cls,
        value,
        spec: FingerprintSpec,
    ) -> "FingerprintProjector":
        """Restore a projection matrix and verify its configured dimensions."""
        matrix = np.frombuffer(
            bytes(value),
            dtype="<f4",
        )

        expected_size = spec.size * spec.projection_dim

        if matrix.size != expected_size:
            raise ContractError(
                "Stored projection dimension mismatch"
            )

        return cls(
            matrix.reshape(
                spec.size,
                spec.projection_dim,
            ).copy()
        )


def _validated_similarity_inputs(
    query,
    candidates,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize and validate query/candidate arrays for chemical similarity."""
    query_vector = np.asarray(
        query,
        dtype=np.float64,
    )
    candidate_matrix = np.asarray(
        candidates,
        dtype=np.float64,
    )

    if (
        query_vector.ndim != 1
        or candidate_matrix.ndim != 2
        or candidate_matrix.shape[1] != query_vector.size
    ):
        raise ValueError("Fingerprint shape mismatch")

    if (
        not np.isfinite(query_vector).all()
        or not np.isfinite(candidate_matrix).all()
        or np.any(query_vector < 0)
        or np.any(candidate_matrix < 0)
    ):
        raise ValueError(
            "Chemical similarities require finite, nonnegative vectors"
        )

    return query_vector, candidate_matrix


def similarity_to_many(
    query,
    candidates,
    metric: str = "minmax",
) -> np.ndarray:
    """Compare one fingerprint with many candidate fingerprints.

    Parameters
    ----------
    query
        One fingerprint vector.
    candidates
        Two-dimensional array with one candidate fingerprint per row.
    metric
        ``"minmax"`` computes generalized Jaccard/min-max similarity, which is
        the default for count fingerprints. ``"tanimoto_dot"`` computes the
        dot-product Tanimoto form and ``"cosine"`` computes cosine similarity.

    Returns
    -------
    numpy.ndarray
        Similarities in ``[0, 1]``, one value per candidate row. Degenerate
        zero denominators evaluate to zero rather than NaN.
    """
    query_vector, candidate_matrix = _validated_similarity_inputs(
        query,
        candidates,
    )

    if metric == "minmax":
        numerator = np.minimum(
            candidate_matrix,
            query_vector,
        ).sum(axis=1)
        denominator = np.maximum(
            candidate_matrix,
            query_vector,
        ).sum(axis=1)

    elif metric == "tanimoto_dot":
        numerator = candidate_matrix @ query_vector
        denominator = (
            np.sum(candidate_matrix * candidate_matrix, axis=1)
            + query_vector @ query_vector
            - numerator
        )

    elif metric == "cosine":
        numerator = candidate_matrix @ query_vector
        denominator = (
            np.linalg.norm(candidate_matrix, axis=1)
            * np.linalg.norm(query_vector)
        )

    else:
        raise ValueError(
            f"Unknown chemical metric: {metric}"
        )

    similarities = np.divide(
        numerator,
        denominator,
        out=np.zeros(len(candidate_matrix), dtype=np.float64),
        where=denominator > 0,
    )

    return np.clip(
        similarities,
        0.0,
        1.0,
    )


def pairwise_similarity(
    vectors,
    metric: str = "minmax",
) -> np.ndarray:
    """Return the full pairwise similarity matrix for fingerprint rows.

    This helper is intended for small candidate sets such as medoid
    aggregation. It deliberately computes the dense square matrix and should
    not be used for all-pairs comparison of the complete reference library.
    """
    matrix = np.asarray(vectors)

    if matrix.ndim != 2:
        raise ValueError(
            "pairwise_similarity requires a two-dimensional fingerprint array"
        )

    if len(matrix) == 0:
        return np.empty(
            (0, 0),
            dtype=np.float64,
        )

    return np.vstack(
        [
            similarity_to_many(
                row,
                matrix,
                metric,
            )
            for row in matrix
        ]
    )

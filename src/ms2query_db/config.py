"""Configuration objects for building and querying MS2Query-DB libraries.

Configuration is intentionally explicit and strict. Unknown settings are
rejected when loading configuration dictionaries or JSON files, which helps
catch misspelled or outdated configuration options early.

A library can contain positive-ion spectra, negative-ion spectra, or both.
When both ion modes are stored, queries are matched to reference spectra of
the same ion mode by default. Cross-mode searching is represented in the
configuration API already, but is not yet implemented by the query engine.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import json
import math
import re


METHODS = (
    "ms2deepscore",
    "cosine",
    "modified_cosine",
    "entropy_fragment",
    "entropy_hybrid",
)

CLASSICAL_METHODS = METHODS[1:]

IONMODES = (
    "positive",
    "negative",
    "both",
)

IONMODE_MATCHING_MODES = (
    "same_mode",
    "cross_mode",
)

CONSENSUS_METHODS = (
    "medoid",
    "centroid",
)

CONSENSUS_WEIGHT_SOURCES = (
    "rf_quality",
    *METHODS,
)

CONSENSUS_SELECTION_METHODS = (
    "top_n",
    "score_delta",
)

CHEMICAL_METRICS = (
    "minmax",
    "tanimoto_dot",
    "cosine",
)

ANNOTATION_POLICIES = (
    "validate_if_present",
    "require",
    "ignore",
)

_SCHEMA_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,47}")


def methods_tuple(
    value: str | Iterable[str] | None,
    *,
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Normalize and validate a collection of spectral search methods.

    Parameters
    ----------
    value
        Search methods as an iterable, a comma-separated string, ``"all"``,
        or ``None``.
    allow_empty
        Whether an empty method collection is permitted.

    Returns
    -------
    tuple[str, ...]
        Validated method names in their original order.

    Raises
    ------
    ValueError
        If no methods are selected when required, a method is unknown, or a
        method is listed more than once.
    """
    if value is None:
        values: tuple[str, ...] = ()
    elif value == "all":
        values = METHODS
    elif isinstance(value, str):
        values = tuple(
            item.strip()
            for item in value.split(",")
            if item.strip()
        )
    else:
        try:
            values = tuple(value)
        except TypeError as exc:
            raise ValueError(
                "Search methods must be an iterable of method names."
            ) from exc

    if not values and not allow_empty:
        raise ValueError("Select at least one search method.")

    if any(not isinstance(item, str) or not item for item in values):
        raise ValueError("Search methods must be non-empty strings.")

    unknown = set(values) - set(METHODS)
    if unknown:
        raise ValueError(
            f"Unknown search methods {sorted(unknown)}; "
            f"choose from {METHODS}."
        )

    if len(set(values)) != len(values):
        raise ValueError("Search methods must not be repeated.")

    return values


def positive_int(value: object, name: str) -> None:
    """Validate that *value* is a positive integer."""
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")


def finite(
    value: object,
    name: str,
    minimum: float | None = 0.0,
) -> None:
    """Validate that *value* is a finite number above an optional minimum."""
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number.")

    try:
        is_finite = math.isfinite(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be a finite number.") from exc

    if not is_finite:
        raise ValueError(f"{name} must be finite.")

    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}.")


def positive_finite(value: object, name: str) -> None:
    """Validate that *value* is a finite number greater than zero."""
    finite(value, name, minimum=None)

    if value <= 0:
        raise ValueError(f"{name} must be > 0.")


def schema_name(value: str) -> str:
    """Validate and return a PostgreSQL application schema name.

    Application schemas are deliberately restricted to short lowercase names
    that do not overlap with PostgreSQL system schemas.
    """
    if not isinstance(value, str) or not _SCHEMA_PATTERN.fullmatch(value):
        raise ValueError(
            "Use a lowercase application schema containing only letters, "
            "digits, and underscores, beginning with a letter and containing "
            "at most 48 characters."
        )

    if (
        value in {"public", "information_schema"}
        or value.startswith("pg_")
    ):
        raise ValueError(
            f"{value!r} is reserved and cannot be used as an application "
            "schema."
        )

    return value


def _validate_optional_column(value: str | None, name: str) -> None:
    """Validate an optional CSV column name."""
    if value is None:
        return

    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string or None.")


def _reject_unknown_keys(
    cls: type,
    value: Mapping[str, object],
) -> None:
    """Reject dictionary keys that are not constructor fields of *cls*."""
    allowed = {
        item.name
        for item in fields(cls)
        if item.init
    }

    unknown = set(value) - allowed

    if unknown:
        raise ValueError(
            f"Unknown {cls.__name__} settings: {sorted(unknown)}."
        )


def _read_json_object(path: str | Path) -> dict[str, object]:
    """Read a JSON configuration file and require a top-level object."""
    path = Path(path)

    data = json.loads(
        path.read_text(encoding="utf-8")
    )

    if not isinstance(data, dict):
        raise ValueError(
            f"Configuration file {path} must contain a JSON object."
        )

    return data


def _json_compatible_dict(instance: object) -> dict[str, object]:
    """Convert a dataclass configuration to a JSON-compatible dictionary."""
    return json.loads(
        json.dumps(asdict(instance))
    )


@dataclass(frozen=True)
class FingerprintSpec:
    """Chemical fingerprint representation used by an MS2Query-DB library.

    The full fingerprint is stored for exact chemical similarity calculations.
    A lower-dimensional deterministic projection is additionally stored in
    pgvector for approximate nearest-neighbour chemical searches.

    Parameters
    ----------
    family
        Fingerprint implementation. Supported values are ``"morgan"`` and
        ``"rdkit"``.
    size
        Number of dimensions in the full fingerprint.
    radius
        Morgan fingerprint radius. This setting is only meaningful for Morgan
        fingerprints.
    count
        Use count fingerprints instead of binary fingerprints.
    value_transform
        Optional transformation applied to count values. ``"none"`` preserves
        raw counts and ``"log1p"`` applies ``log(1 + count)``.
    include_chirality
        Include chirality in Morgan fingerprints.
    projection_dim
        Dimensionality of the deterministic vector used for pgvector chemical
        ANN search.
    projection_seed
        Seed controlling the deterministic fingerprint projection.
    n_jobs
        Number of worker processes used when computing fingerprints.
    """

    family: str = "morgan"
    size: int = 4096
    radius: int = 9
    count: bool = True
    value_transform: str = "none"
    include_chirality: bool = False
    projection_dim: int = 256
    projection_seed: int = 56789
    n_jobs: int = 1

    def __post_init__(self) -> None:
        if self.family not in {"morgan", "rdkit"}:
            raise ValueError(
                "Fingerprint family must be 'morgan' or 'rdkit'."
            )

        for name in (
            "size",
            "projection_dim",
            "n_jobs",
        ):
            positive_int(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.radius, bool)
            or not isinstance(self.radius, int)
            or self.radius < 0
        ):
            raise ValueError(
                "radius must be a nonnegative integer."
            )

        if not isinstance(self.count, bool):
            raise ValueError("count must be a boolean.")

        if not isinstance(self.include_chirality, bool):
            raise ValueError(
                "include_chirality must be a boolean."
            )

        if (
            self.family == "rdkit"
            and self.include_chirality
        ):
            raise ValueError(
                "include_chirality is only supported for Morgan fingerprints."
            )

        if (
            isinstance(self.projection_seed, bool)
            or not isinstance(self.projection_seed, int)
            or self.projection_seed < 0
        ):
            raise ValueError(
                "projection_seed must be a nonnegative integer."
            )

        if self.projection_dim > 2000:
            raise ValueError(
                "Projected pgvector HNSW vectors support at most "
                "2000 dimensions."
            )

        if self.value_transform not in {
            "none",
            "log1p",
        }:
            raise ValueError(
                "value_transform must be 'none' or 'log1p'."
            )

        if (
            self.value_transform != "none"
            and not self.count
        ):
            raise ValueError(
                "value_transform='log1p' requires count=True."
            )


@dataclass(frozen=True)
class ProcessingSpec:
    """Preprocessing settings applied consistently to MS/MS spectra.

    These settings define the peak range and intensity filtering used before
    spectra enter spectral similarity indexes or query calculations.

    Parameters
    ----------
    mz_min, mz_max
        Inclusive fragment m/z range retained during preprocessing.
    min_relative_intensity
        Remove peaks whose intensity is below this fraction of the largest
        peak intensity.
    min_peaks
        Minimum number of retained peaks required for a usable spectrum.
    max_peaks
        Maximum number of retained peaks.
    """

    mz_min: float = 10.0
    mz_max: float = 1000.0
    min_relative_intensity: float = 0.001
    min_peaks: int = 5
    max_peaks: int = 1000

    def __post_init__(self) -> None:
        finite(self.mz_min, "mz_min")
        finite(self.mz_max, "mz_max")

        if self.mz_max <= self.mz_min:
            raise ValueError(
                "mz_max must exceed mz_min."
            )

        finite(
            self.min_relative_intensity,
            "min_relative_intensity",
        )

        if not 0 <= self.min_relative_intensity <= 1:
            raise ValueError(
                "min_relative_intensity must be in [0, 1]."
            )

        positive_int(
            self.min_peaks,
            "min_peaks",
        )
        positive_int(
            self.max_peaks,
            "max_peaks",
        )

        if self.max_peaks < self.min_peaks:
            raise ValueError(
                "max_peaks must be >= min_peaks."
            )


@dataclass(frozen=True)
class ScoringSpec:
    """Parameters shared by native classical spectrum similarities.

    Parameters
    ----------
    tolerance
        Fragment m/z matching tolerance in Da.
    noise_cutoff
        Relative intensity threshold used by native Flash-style similarities.
    intensity_power
        Exponent applied to peak intensities before comparison.
    remove_precursor
        Remove fragment peaks close to the precursor before scoring.
    offset_to_precursor
        Upper fragment-mass offset relative to the precursor when precursor
        removal is enabled.
    dtype
        Floating-point precision used by native indexes.
    """

    tolerance: float = 0.1
    noise_cutoff: float = 0.01
    intensity_power: float = 1.0
    remove_precursor: bool = True
    offset_to_precursor: float = -1.6
    dtype: str = "float64"

    def __post_init__(self) -> None:
        positive_finite(
            self.tolerance,
            "tolerance",
        )

        finite(
            self.intensity_power,
            "intensity_power",
        )

        finite(
            self.noise_cutoff,
            "noise_cutoff",
        )

        if not 0 <= self.noise_cutoff <= 1:
            raise ValueError(
                "noise_cutoff must be in [0, 1]."
            )

        if not isinstance(
            self.remove_precursor,
            bool,
        ):
            raise ValueError(
                "remove_precursor must be a boolean."
            )

        finite(
            self.offset_to_precursor,
            "offset_to_precursor",
            minimum=None,
        )

        if self.dtype not in {
            "float32",
            "float64",
        }:
            raise ValueError(
                "dtype must be 'float32' or 'float64'."
            )


@dataclass(frozen=True)
class BuildConfig:
    """Configuration for constructing an MS2Query-DB reference library.

    A library may contain one ion mode or both ion modes.

    ``ionmode="positive"`` and ``ionmode="negative"`` build a single-mode
    library. ``ionmode="both"`` retains positive and negative reference
    spectra in the same logical library. Each spectrum must then carry valid
    ion-mode metadata so that same-mode querying can route positive queries to
    positive references and negative queries to negative references.

    Classical search indexes are created only after reference import succeeds.

    Parameters
    ----------
    schema
        PostgreSQL schema used by this library.
    ionmode
        Reference-library ion-mode content: ``"positive"``, ``"negative"``,
        or ``"both"``.
    batch_size
        Number of records handled per import transaction batch.
    fingerprints
        Chemical fingerprint configuration.
    query_processing
        Spectrum preprocessing configuration shared by references and queries.
    scoring
        Classical spectral-similarity settings.
    classical_methods
        Classical search indexes built after import. MS2DeepScore uses
        pgvector and therefore is not included here.
    index_shard_spectra
        Maximum number of spectra per on-disk classical-search shard.
    index_shard_peaks
        Approximate maximum number of peaks per classical-search shard.
    mz_precision
        Peak m/z precision used when representing spectra internally.
    recursive
        Search reference directories recursively for input files.
    pattern
        File glob used to locate reference spectra.
    smiles_column, inchi_column, inchikey_column
        Chemical metadata column names.
    csv_delimiter
        Delimiter used by the compound metadata file.
    invalid_policy
        Whether invalid input records are skipped or immediately raise.
    hnsw_m, hnsw_ef_construction
        pgvector HNSW construction parameters.
    maintenance_work_mem_mb
        PostgreSQL maintenance memory requested while building indexes.
    """

    schema: str = "ms2query_v1"
    ionmode: str = "positive"
    batch_size: int = 1000

    fingerprints: FingerprintSpec = field(
        default_factory=FingerprintSpec
    )

    query_processing: ProcessingSpec = field(
        default_factory=ProcessingSpec
    )

    scoring: ScoringSpec = field(
        default_factory=ScoringSpec
    )

    classical_methods: tuple[str, ...] = (
        "entropy_hybrid",
    )

    index_shard_spectra: int = 20_000
    index_shard_peaks: int = 1_000_000

    mz_precision: float = 1e-6

    recursive: bool = False
    pattern: str = "*.mgf"

    smiles_column: str = "smiles"
    inchi_column: str | None = "inchi"
    inchikey_column: str | None = "inchikey"
    csv_delimiter: str = ","

    invalid_policy: str = "skip"

    hnsw_m: int = 16
    hnsw_ef_construction: int = 100
    maintenance_work_mem_mb: int = 256

    def __post_init__(self) -> None:
        schema_name(self.schema)

        if self.ionmode not in IONMODES:
            raise ValueError(
                f"ionmode must be one of {IONMODES}."
            )

        for name in (
            "batch_size",
            "index_shard_spectra",
            "index_shard_peaks",
            "maintenance_work_mem_mb",
        ):
            positive_int(
                getattr(self, name),
                name,
            )

        if (
            isinstance(self.hnsw_m, bool)
            or not isinstance(self.hnsw_m, int)
            or not 2 <= self.hnsw_m <= 100
        ):
            raise ValueError(
                "hnsw_m must be an integer between 2 and 100."
            )

        minimum_ef = max(
            4,
            2 * self.hnsw_m,
        )

        if (
            isinstance(self.hnsw_ef_construction, bool)
            or not isinstance(
                self.hnsw_ef_construction,
                int,
            )
            or not minimum_ef
            <= self.hnsw_ef_construction
            <= 1000
        ):
            raise ValueError(
                "hnsw_ef_construction must be between "
                f"{minimum_ef} and 1000 for hnsw_m={self.hnsw_m}."
            )

        if self.invalid_policy not in {
            "skip",
            "raise",
        }:
            raise ValueError(
                "invalid_policy must be 'skip' or 'raise'."
            )

        if (
            not isinstance(self.recursive, bool)
        ):
            raise ValueError(
                "recursive must be a boolean."
            )

        if (
            not isinstance(self.pattern, str)
            or not self.pattern.strip()
        ):
            raise ValueError(
                "pattern must be a non-empty string."
            )

        if (
            not isinstance(self.csv_delimiter, str)
            or len(self.csv_delimiter) != 1
        ):
            raise ValueError(
                "csv_delimiter must contain exactly one character."
            )

        if (
            not isinstance(self.smiles_column, str)
            or not self.smiles_column.strip()
        ):
            raise ValueError(
                "smiles_column must be a non-empty string."
            )

        _validate_optional_column(
            self.inchi_column,
            "inchi_column",
        )
        _validate_optional_column(
            self.inchikey_column,
            "inchikey_column",
        )

        positive_finite(
            self.mz_precision,
            "mz_precision",
        )

        exponent = -math.log10(
            self.mz_precision
        )

        if (
            self.mz_precision > 1
            or self.mz_precision < 1e-12
            or not math.isclose(
                exponent,
                round(exponent),
                abs_tol=1e-9,
            )
        ):
            raise ValueError(
                "mz_precision must be a power of ten "
                "between 1e-12 and 1."
            )

        classical_methods = methods_tuple(
            self.classical_methods,
            allow_empty=True,
        )

        non_classical = (
            set(classical_methods)
            - set(CLASSICAL_METHODS)
        )

        if non_classical:
            raise ValueError(
                "classical_methods may contain only classical "
                f"search methods {CLASSICAL_METHODS}; got "
                f"{sorted(non_classical)}."
            )

        object.__setattr__(
            self,
            "classical_methods",
            classical_methods,
        )

    def to_dict(self) -> dict[str, object]:
        """Return this configuration as a JSON-compatible dictionary."""
        return _json_compatible_dict(self)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> BuildConfig:
        """Construct a build configuration from a strict mapping."""
        if not isinstance(value, Mapping):
            raise ValueError(
                "Build configuration must be a mapping."
            )

        data = dict(value)
        _reject_unknown_keys(cls, data)

        nested = (
            ("fingerprints", FingerprintSpec),
            ("query_processing", ProcessingSpec),
            ("scoring", ScoringSpec),
        )

        for key, kind in nested:
            if key not in data:
                continue

            nested_value = data[key]

            if not isinstance(
                nested_value,
                Mapping,
            ):
                raise ValueError(
                    f"{key} must be a configuration object."
                )

            nested_data = dict(nested_value)
            _reject_unknown_keys(
                kind,
                nested_data,
            )

            data[key] = kind(
                **nested_data
            )

        return cls(**data)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> BuildConfig:
        """Load a build configuration from a JSON file."""
        return cls.from_dict(
            _read_json_object(path)
        )


@dataclass(frozen=True)
class QueryConfig:
    """Configuration controlling spectral search and candidate aggregation.

    Search methods independently retrieve their top candidates. When multiple
    methods are enabled, candidate rows are deduplicated into a union and
    complementary scores can be calculated for candidates not originally
    retrieved by a method.

    Ion-mode handling is controlled independently from library construction.
    ``same_mode`` is the normal behavior: positive queries search positive
    references and negative queries search negative references. ``cross_mode``
    is reserved for future support of comparisons across ion modes.

    Consensus is optional. Setting ``consensus_method=None`` returns ordinary
    candidate-ranking results. ``"medoid"`` selects a real candidate compound
    that is chemically central among highly supported candidates.
    ``"centroid"`` constructs a weighted fingerprint and performs a chemical
    database search.

    The current recommended medoid weighting uses RF-predicted candidate
    quality raised to a power of two.

    Parameters
    ----------
    search_methods
        Spectral methods used for candidate retrieval.
    top_n
        Number of candidates retrieved independently by each search method.
    additional_top_n_scores
        Methods that do not retrieve candidates but whose scores should be
        calculated for the retrieved candidate union.
    batch_size
        Number of query spectra handled together.
    classical_query_batch
        Number of queries passed to a native classical index in one call.
    classical_n_jobs
        Worker count used by classical search.
    ef_search
        pgvector HNSW search breadth for embedding retrieval.
    exact_embedding_search
        Use exact embedding search instead of pgvector HNSW.
    ionmode_matching
        ``"same_mode"`` for mode-matched search or ``"cross_mode"`` for the
        future cross-ion-mode search mode.
    annotation_policy
        Handling of structural annotations supplied with query spectra.
    compute_chemical_similarity
        Compute ground-truth chemical similarity when query structure is
        available. Primarily useful for evaluation.
    chemical_metric
        Chemical fingerprint similarity metric.
    consensus_method
        ``None``, ``"medoid"``, or ``"centroid"``.
    consensus_weight_source
        Candidate score used for consensus weighting. ``"rf_quality"`` is the
        recommended source when an RF reranker is available.
    consensus_weight_power
        Exponent applied to non-negative candidate weights before consensus.
        The current default is 2.0.
    consensus_selection
        Candidate subset rule used before consensus. ``"top_n"`` keeps the
        highest-scoring candidates and ``"score_delta"`` retains candidates
        within ``consensus_score_delta`` of the best score.
    consensus_top_n
        Maximum candidate count for top-N consensus.
    consensus_score_delta
        Maximum score difference from the best candidate when using
        ``consensus_selection="score_delta"``.
    consensus_pool
        Chemical ANN pool size used by centroid/database consensus.
    exact_chemical_search
        Use exact full-library chemical search instead of chemical ANN.
    """

    search_methods: tuple[str, ...] = (
        "ms2deepscore",
        "entropy_hybrid",
    )

    top_n: int = 20

    additional_top_n_scores: (
        tuple[str, ...] | None
    ) = None

    batch_size: int = 32
    classical_query_batch: int = 32
    classical_n_jobs: int = 1

    ef_search: int = 200
    exact_embedding_search: bool = False

    ionmode_matching: str = "same_mode"

    annotation_policy: str = (
        "validate_if_present"
    )

    compute_chemical_similarity: bool = False
    chemical_metric: str = "minmax"

    consensus_method: str | None = None
    consensus_weight_source: str = "rf_quality"
    consensus_weight_power: float = 2.0

    consensus_selection: str = "top_n"
    consensus_top_n: int = 20
    consensus_score_delta: float = 0.2

    consensus_pool: int = 500
    exact_chemical_search: bool = False

    def __post_init__(self) -> None:
        search_methods = methods_tuple(
            self.search_methods
        )

        object.__setattr__(
            self,
            "search_methods",
            search_methods,
        )

        extra = methods_tuple(
            self.additional_top_n_scores,
            allow_empty=True,
        )

        overlap = (
            set(search_methods)
            & set(extra)
        )

        if overlap:
            raise ValueError(
                "additional_top_n_scores must not repeat retrieval "
                f"methods; duplicated methods: {sorted(overlap)}."
            )

        object.__setattr__(
            self,
            "additional_top_n_scores",
            extra or None,
        )

        for name in (
            "top_n",
            "batch_size",
            "classical_query_batch",
            "classical_n_jobs",
            "ef_search",
            "consensus_top_n",
            "consensus_pool",
        ):
            positive_int(
                getattr(self, name),
                name,
            )

        if self.ef_search > 1000:
            raise ValueError(
                "pgvector hnsw.ef_search must be <= 1000."
            )

        if not isinstance(
            self.exact_embedding_search,
            bool,
        ):
            raise ValueError(
                "exact_embedding_search must be a boolean."
            )

        if self.ionmode_matching not in (
            IONMODE_MATCHING_MODES
        ):
            raise ValueError(
                "ionmode_matching must be one of "
                f"{IONMODE_MATCHING_MODES}."
            )

        if self.annotation_policy not in (
            ANNOTATION_POLICIES
        ):
            raise ValueError(
                "annotation_policy must be one of "
                f"{ANNOTATION_POLICIES}."
            )

        if not isinstance(
            self.compute_chemical_similarity,
            bool,
        ):
            raise ValueError(
                "compute_chemical_similarity must be a boolean."
            )

        if self.chemical_metric not in (
            CHEMICAL_METRICS
        ):
            raise ValueError(
                "chemical_metric must be one of "
                f"{CHEMICAL_METRICS}."
            )

        if (
            self.consensus_method is not None
            and self.consensus_method
            not in CONSENSUS_METHODS
        ):
            raise ValueError(
                "consensus_method must be None or one of "
                f"{CONSENSUS_METHODS}."
            )

        if (
            self.consensus_weight_source
            not in CONSENSUS_WEIGHT_SOURCES
        ):
            raise ValueError(
                "consensus_weight_source must be one of "
                f"{CONSENSUS_WEIGHT_SOURCES}."
            )

        positive_finite(
            self.consensus_weight_power,
            "consensus_weight_power",
        )

        if (
            self.consensus_selection
            not in CONSENSUS_SELECTION_METHODS
        ):
            raise ValueError(
                "consensus_selection must be one of "
                f"{CONSENSUS_SELECTION_METHODS}."
            )

        finite(
            self.consensus_score_delta,
            "consensus_score_delta",
        )

        available_scores = (
            set(search_methods)
            | set(extra)
        )

        if (
            self.consensus_method is not None
            and self.consensus_weight_source in METHODS
            and self.consensus_weight_source
            not in available_scores
        ):
            raise ValueError(
                "The spectral consensus_weight_source must either "
                "retrieve candidates or be listed in "
                "additional_top_n_scores."
            )

        if (
            self.consensus_method == "centroid"
            and self.consensus_pool
            < self.consensus_top_n
        ):
            raise ValueError(
                "consensus_pool must be >= consensus_top_n for "
                "centroid consensus."
            )

        if not isinstance(
            self.exact_chemical_search,
            bool,
        ):
            raise ValueError(
                "exact_chemical_search must be a boolean."
            )

    def to_dict(self) -> dict[str, object]:
        """Return this configuration as a JSON-compatible dictionary."""
        return _json_compatible_dict(self)

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> QueryConfig:
        """Construct a query configuration from a strict mapping."""
        if not isinstance(value, Mapping):
            raise ValueError(
                "Query configuration must be a mapping."
            )

        data = dict(value)
        _reject_unknown_keys(
            cls,
            data,
        )

        return cls(**data)

    @classmethod
    def from_file(
        cls,
        path: str | Path,
    ) -> QueryConfig:
        """Load a query configuration from a JSON file."""
        return cls.from_dict(
            _read_json_object(path)
        )
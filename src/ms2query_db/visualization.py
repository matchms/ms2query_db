"""Chemical-space visualization coordinates derived from reference compounds.

The public operation in this module is deliberately library-level: UMAP coordinates
are meaningful only when they are fitted to the complete compound collection.  The
result is therefore computed once from all stored reference compounds and then
written back to ``reference_compounds.visual_x``/``visual_y`` as derived metadata.
"""
from __future__ import annotations

import inspect
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .errors import CompatibilityError, ContractError

_VISUAL_COLUMNS = (
    "compound_id",
    "identity_key",
    "inchikey14",
    "inchikey",
    "smiles",
    "inchi",
    "formula",
    "exact_mass",
    "superclass",
    "class",
    "subclass",
    "visual_x",
    "visual_y",
    "pubchem_cid",
)


def _all_compounds(database, *, batch_size: int) -> pd.DataFrame:
    """Materialize lightweight compound metadata in stable database-ID order."""
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")

    rows: list[dict[str, Any]] = []
    for block in database.iter_visualization_compounds(batch_size=batch_size):
        rows.extend(block)

    if not rows:
        raise ValueError("The library contains no reference compounds to visualize.")

    frame = pd.DataFrame(rows)
    if "compound_class" in frame.columns:
        frame = frame.rename(columns={"compound_class": "class"})

    required = {"compound_id", "smiles"}
    missing = required - set(frame.columns)
    if missing:
        raise ContractError(
            "Database visualization rows are missing required columns: "
            f"{sorted(missing)}."
        )

    if frame["compound_id"].duplicated().any():
        raise ContractError("Database returned duplicate compound IDs for visualization.")
    if frame["smiles"].isna().any() or frame["smiles"].astype(str).str.len().eq(0).any():
        raise ContractError("Reference compounds contain missing canonical SMILES.")

    return frame.sort_values("compound_id", kind="stable").reset_index(drop=True)


def _rdkit_logcount_generator(*, fp_size: int):
    """Return the folded RDKit path-fingerprint generator used for log-count UMAP."""
    if isinstance(fp_size, bool) or not isinstance(fp_size, int) or fp_size < 1:
        raise ValueError("fp_size must be a positive integer.")

    from rdkit.Chem import rdFingerprintGenerator

    return rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)


def _supported_argument(parameters, *names: str) -> str | None:
    for name in names:
        if name in parameters:
            return name
    return None


def _chemap_logcount_kwargs(function, *, fpgen) -> dict[str, Any]:
    """Build fingerprint kwargs for a chemap UMAP helper without silent downgrade.

    Chemap's plotting API has evolved while the project has been young.  We inspect
    the installed callable so MS2Query-DB can use either a dedicated fingerprint
    configuration argument or explicit count/transform arguments.  Crucially, this
    helper refuses to silently fall back to binary fingerprints: the requested
    representation is folded RDKit *log-count* fingerprints.
    """
    parameters = inspect.signature(function).parameters
    kwargs: dict[str, Any] = {}

    fpgen_name = _supported_argument(parameters, "fpgen", "fingerprint_generator")
    if fpgen_name is None:
        raise CompatibilityError(
            "Installed chemap UMAP helper does not accept an RDKit fingerprint "
            "generator. Upgrade chemap to a version exposing fpgen=... ."
        )
    kwargs[fpgen_name] = fpgen

    config_name = _supported_argument(
        parameters,
        "fp_config",
        "fingerprint_config",
    )
    if config_name is not None:
        from chemap import FingerprintConfig

        kwargs[config_name] = FingerprintConfig(
            count=True,
            folded=True,
            return_csr=False,
            invalid_policy="raise",
        )
    else:
        count_name = _supported_argument(
            parameters,
            "count",
            "fp_count",
            "fingerprint_count",
        )
        if count_name is not None:
            kwargs[count_name] = True

        folded_name = _supported_argument(
            parameters,
            "folded",
            "fp_folded",
            "fingerprint_folded",
        )
        if folded_name is not None:
            kwargs[folded_name] = True

    transform_name = _supported_argument(
        parameters,
        "value_transform",
        "fp_value_transform",
        "fingerprint_value_transform",
        "transform",
    )
    log_flag_name = _supported_argument(
        parameters,
        "logcount",
        "log_count",
        "log_counts",
    )

    if transform_name is not None:
        kwargs[transform_name] = "log1p"
    elif log_flag_name is not None:
        kwargs[log_flag_name] = True
    else:
        doc = (inspect.getdoc(function) or "").lower()
        # Some chemap releases define log-count as the plotting helper's fixed
        # representation and therefore expose no transform parameter.  Only trust
        # that behavior when the callable documents it explicitly.
        if not ("log" in doc and "count" in doc):
            raise CompatibilityError(
                "Installed chemap UMAP helper does not expose or document a "
                "log-count fingerprint transform. MS2Query-DB will not silently "
                "compute different visualization fingerprints; upgrade chemap."
            )

    return kwargs


def _run_chemap_umap(
    compounds: pd.DataFrame,
    *,
    backend: str,
    fp_size: int,
    random_state: int | None,
    umap_kwargs: dict[str, Any] | None,
    umap_function=None,
) -> pd.DataFrame:
    """Run the selected chemap UMAP helper and validate row-preserving output."""
    if backend not in {"gpu", "cpu"}:
        raise ValueError("backend must be 'gpu' or 'cpu'.")

    if umap_function is None:
        from chemap.plotting import (
            create_chem_space_umap,
            create_chem_space_umap_gpu,
        )

        function = (
            create_chem_space_umap_gpu
            if backend == "gpu"
            else create_chem_space_umap
        )
    else:
        function = umap_function

    fpgen = _rdkit_logcount_generator(fp_size=fp_size)
    parameters = inspect.signature(function).parameters
    kwargs: dict[str, Any] = {
        "col_smiles": "smiles",
        "inplace": False,
        "x_col": "visual_x",
        "y_col": "visual_y",
    }
    kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in parameters
    }
    kwargs.update(_chemap_logcount_kwargs(function, fpgen=fpgen))

    if "metric" in parameters:
        # GPU chemap currently supports cosine for this path.  Keeping the same
        # default for CPU makes the two backends represent the same geometry.
        kwargs["metric"] = "cosine"
    if random_state is not None and "random_state" in parameters:
        kwargs["random_state"] = int(random_state)

    for key, value in (umap_kwargs or {}).items():
        if key not in parameters:
            raise TypeError(
                f"Installed chemap {function.__name__}() does not accept {key!r}."
            )
        if key in {
            "col_smiles",
            "inplace",
            "x_col",
            "y_col",
            "fpgen",
            "fingerprint_generator",
            "metric",
        }:
            raise ValueError(f"UMAP argument {key!r} is managed by MS2Query-DB.")
        kwargs[key] = value

    result = function(compounds.copy(), **kwargs)
    if not isinstance(result, pd.DataFrame):
        raise ContractError(
            f"Chemap {function.__name__}() returned {type(result).__name__}; "
            "expected a pandas DataFrame."
        )
    if len(result) != len(compounds):
        raise ContractError(
            "Chemap changed compound row count while computing visualization "
            f"coordinates: {len(compounds)} -> {len(result)}."
        )

    if "compound_id" not in result.columns:
        raise ContractError("Chemap output lost compound_id row provenance.")

    expected_ids = compounds["compound_id"].to_numpy(dtype=np.int64)
    actual_ids = result["compound_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(actual_ids, expected_ids):
        raise ContractError(
            "Chemap changed compound row order while computing visualization coordinates."
        )

    if "visual_x" not in result.columns or "visual_y" not in result.columns:
        raise ContractError(
            "Chemap output does not contain the requested visual_x/visual_y columns."
        )

    coordinates = result[["visual_x", "visual_y"]].to_numpy(dtype=np.float64)
    if coordinates.shape != (len(compounds), 2) or not np.isfinite(coordinates).all():
        raise ContractError("Chemap produced missing or nonfinite UMAP coordinates.")

    output = result.copy()
    output["visual_x"] = coordinates[:, 0]
    output["visual_y"] = coordinates[:, 1]
    return output


def _export_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    """Atomically export visualization rows as a CSV file."""
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    suffix = ".tmp"
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        suffix=suffix,
        prefix=f".{output.name}.",
        dir=output.parent,
        delete=False,
    )
    temporary = Path(handle.name)
    handle.close()
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return output


def compute_library_visual_coordinates(
    database,
    *,
    output_csv: str | Path | None = None,
    backend: str = "gpu",
    overwrite: bool = False,
    fp_size: int = 4096,
    random_state: int | None = 42,
    batch_size: int = 5000,
    umap_kwargs: dict[str, Any] | None = None,
    umap_function=None,
) -> pd.DataFrame:
    """Compute, persist, optionally export, and return chemical-space coordinates.

    Parameters
    ----------
    database
        Open :class:`~ms2query_db.db.Database` for a ready library.
    output_csv
        Optional path receiving one CSV row per reference compound.
    backend
        ``"gpu"`` (default) uses chemap's ``create_chem_space_umap_gpu``;
        ``"cpu"`` uses ``create_chem_space_umap``.
    overwrite
        Replace already populated ``visual_x``/``visual_y`` values.  The default
        is ``False`` because these are intended as one-time derived library data.
    fp_size
        Folded RDKit path-fingerprint size.  The representation is count-based
        with a ``log1p`` transform and cosine UMAP distance.
    random_state
        Reproducibility seed when supported by the installed chemap helper.
    batch_size
        Database read batch size. UMAP still materializes all lightweight
        compound rows in memory because it is fitted globally.
    umap_kwargs
        Additional keyword arguments accepted by the selected chemap helper,
        for example ``n_neighbors`` or ``min_dist``.

    Returns
    -------
    pandas.DataFrame
        Compound metadata plus finite ``visual_x`` and ``visual_y`` columns.
    """
    if not isinstance(overwrite, bool):
        raise TypeError("overwrite must be a boolean.")

    # Coordinate fitting and persistence are one library-wide writer operation.
    # The advisory lock also prevents two expensive UMAP jobs from racing and
    # letting the last writer silently replace the first result.
    with database.build_lock():
        compounds = _all_compounds(database, batch_size=batch_size)

        if not overwrite:
            already_visualized = compounds[
                compounds["visual_x"].notna()
                | compounds["visual_y"].notna()
            ]
            if not already_visualized.empty:
                preview = already_visualized["compound_id"].tolist()[:10]
                raise FileExistsError(
                    "Visualization coordinates already exist for compound IDs "
                    f"{preview}; pass overwrite=True to recompute the complete "
                    "library."
                )

        embedded = _run_chemap_umap(
            compounds,
            backend=backend,
            fp_size=fp_size,
            random_state=random_state,
            umap_kwargs=umap_kwargs,
            umap_function=umap_function,
        )

        database.update_visual_coordinates(
            embedded[["compound_id", "visual_x", "visual_y"]].to_dict("records"),
            overwrite=overwrite,
        )

    columns = [column for column in _VISUAL_COLUMNS if column in embedded.columns]
    result = embedded.reindex(columns=columns).copy()

    if output_csv is not None:
        _export_csv(result, output_csv)

    return result

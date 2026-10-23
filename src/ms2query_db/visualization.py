"""Library-wide chemical-space coordinates for MS2Query-DB compounds."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sqlalchemy import bindparam, exists, select, update

from .config import BuildConfig
from .db import Database
from .errors import ContractError
from .util import chunks

VisualSelection = Literal["references", "all"] | Sequence[str]


def _normalize_selection(selection: VisualSelection) -> tuple[str, tuple[str, ...] | None]:
    """Return ``(mode, classes)`` for the public selection argument."""
    if isinstance(selection, str):
        if selection not in {"references", "all"}:
            raise ValueError(
                "selection must be 'references', 'all', or a non-empty sequence "
                "of compound class names."
            )
        return selection, None

    if not isinstance(selection, Sequence):
        raise TypeError(
            "selection must be 'references', 'all', or a sequence of class names."
        )

    classes = tuple(dict.fromkeys(selection))
    if not classes or any(not isinstance(item, str) or not item.strip() for item in classes):
        raise ValueError("Class selection must contain non-empty strings.")

    return "classes", tuple(item.strip() for item in classes)


def _selected_compounds(
    database: Database,
    selection: VisualSelection,
) -> pd.DataFrame:
    """Load only the compound metadata needed for one UMAP calculation."""
    mode, classes = _normalize_selection(selection)

    if mode == "references":
        rows = database.get_visualization_compounds(
            reference_only=True,
        )
    elif mode == "classes":
        rows = database.get_visualization_compounds(
            classes=classes,
        )
    else:
        rows = database.get_visualization_compounds()

    frame = pd.DataFrame(rows)

    if frame.empty:
        raise ValueError("The selected compound set is empty.")

    return frame


def _validate_coordinates(before: pd.DataFrame, after: pd.DataFrame) -> None:
    """Require Chemap to preserve compound rows and return finite coordinates."""
    if len(after) != len(before):
        raise ContractError("Chemap changed the compound row count during UMAP.")

    if "compound_id" not in after or not np.array_equal(
        after["compound_id"].to_numpy(),
        before["compound_id"].to_numpy(),
    ):
        raise ContractError("Chemap changed compound row alignment during UMAP.")

    coordinates = after[["visual_x", "visual_y"]].to_numpy(dtype=np.float64)
    if coordinates.shape != (len(before), 2) or not np.isfinite(coordinates).all():
        raise ContractError("Chemap returned invalid visual_x/visual_y coordinates.")


def compute_visual_coordinates(
    database_url: str,
    *,
    schema: str | None = None,
    selection: VisualSelection = "references",
    output_csv: str | Path | None = None,
    update_database: bool = True,
    overwrite: bool = False,
    gpu: bool = True,
    fp_size: int = 4096,
    n_neighbors: int = 100,
    min_dist: float = 0.25,
    random_state: int | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Compute and optionally persist 2D chemical-space coordinates.

    Parameters
    ----------
    database_url:
        PostgreSQL connection URL for the MS2Query-DB library.
    schema:
        Library schema. Defaults to :class:`BuildConfig`'s schema.
    selection:
        Which compounds to map:

        - ``"references"``: compounds that have at least one reference spectrum;
        - ``"all"``: every compound in ``reference_compounds``;
        - sequence of strings: compounds whose ``compound_class`` exactly matches
          one of the supplied class names.
    output_csv:
        Optional CSV path. The exported rows are exactly the compounds used for
        the UMAP calculation, including ``compound_id``, structure/class metadata,
        and the newly computed ``visual_x``/``visual_y`` values.
    update_database:
        If true, write the coordinates to ``reference_compounds``.
    overwrite:
        If false, database writing fails when any selected row already has a
        visual coordinate. If true, coordinates for all selected rows are
        replaced. This flag has no effect when ``update_database=False``.
    gpu:
        Use Chemap's cuML GPU UMAP implementation. If false, use the CPU UMAP
        implementation with cosine distance so CPU and GPU runs use the same
        fingerprint representation and distance family.
    fp_size:
        Size of the folded RDKit path fingerprint.
    n_neighbors, min_dist, random_state:
        UMAP settings forwarded to Chemap.
    show_progress:
        Forwarded to Chemap fingerprint computation.

    Returns
    -------
    pandas.DataFrame
        The selected compounds with newly computed ``visual_x`` and ``visual_y``.

    Notes
    -----
    The chemical-space representation is intentionally fixed here to a folded
    RDKit count fingerprint followed by ``log1p`` scaling and cosine UMAP. This
    keeps the MS2Query-DB visualization definition small and reproducible while
    leaving lower-level fingerprint/UMAP flexibility in Chemap itself.
    """
    if isinstance(fp_size, bool) or not isinstance(fp_size, int) or fp_size < 1:
        raise ValueError("fp_size must be a positive integer.")

    from chemap import FingerprintConfig
    from chemap.plotting import create_chem_space_umap, create_chem_space_umap_gpu
    from rdkit.Chem import rdFingerprintGenerator

    database = Database(database_url, schema or BuildConfig().schema)
    try:
        database.open()
        compounds = _selected_compounds(database, selection)

        # Fail before the expensive UMAP calculation when a non-overwrite DB
        # write cannot succeed anyway.
        if update_database and not overwrite:
            existing = compounds["visual_x"].notna() | compounds["visual_y"].notna()
            if existing.any():
                raise FileExistsError(
                    f"{int(existing.sum())} selected compound(s) already have visual "
                    "coordinates; pass overwrite=True to replace them."
                )

        fpgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=fp_size)
        fingerprint_config = FingerprintConfig(
            count=True,
            folded=True,
            return_csr=False,
            invalid_policy="raise",
        )

        kwargs = {
            "col_smiles": "smiles",
            "inplace": False,
            "x_col": "visual_x",
            "y_col": "visual_y",
            "fpgen": fpgen,
            "fingerprint_config": fingerprint_config,
            "show_progress": show_progress,
            "scaling": "log1p",
            "n_neighbors": n_neighbors,
            "min_dist": min_dist,
            "umap_random_state": random_state,
        }

        if gpu:
            result = create_chem_space_umap_gpu(compounds, **kwargs)
        else:
            result = create_chem_space_umap(
                compounds,
                distance_function="cosine",
                n_jobs=-1,
                **kwargs,
            )

        _validate_coordinates(compounds, result)

        if update_database:
            database.update_visual_coordinates(
                result[
                    ["compound_id", "visual_x", "visual_y"]
                ].to_dict("records"),
                overwrite=overwrite,
            )

        if output_csv is not None:
            path = Path(output_csv).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            result.to_csv(path, index=False)

        return result
    finally:
        database.close()

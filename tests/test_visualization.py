from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# Adjust this import if the project chooses another module name.
from ms2query_db.visualization import (
    _normalize_selection,
    _validate_coordinates,
)


def test_normalize_reference_selection():
    assert _normalize_selection("references") == ("references", None)


def test_normalize_all_selection():
    assert _normalize_selection("all") == ("all", None)


def test_normalize_class_selection_deduplicates_in_order():
    assert _normalize_selection(["Lipids", "Alkaloids", "Lipids"]) == (
        "classes",
        ("Lipids", "Alkaloids"),
    )


@pytest.mark.parametrize("selection", ["Lipids", "", "reference"])
def test_normalize_rejects_unknown_string_modes(selection):
    with pytest.raises(ValueError):
        _normalize_selection(selection)


@pytest.mark.parametrize("selection", [[], ["Lipids", ""], ["Lipids", None]])
def test_normalize_rejects_invalid_class_lists(selection):
    with pytest.raises(ValueError):
        _normalize_selection(selection)


def test_validate_coordinates_accepts_aligned_finite_result():
    before = pd.DataFrame(
        {
            "compound_id": [1, 2],
            "visual_x": [None, None],
            "visual_y": [None, None],
        }
    )
    after = pd.DataFrame(
        {
            "compound_id": [1, 2],
            "visual_x": [0.1, 0.2],
            "visual_y": [0.3, 0.4],
        }
    )

    _validate_coordinates(before, after)


def test_validate_coordinates_rejects_changed_alignment():
    before = pd.DataFrame({"compound_id": [1, 2]})
    after = pd.DataFrame(
        {
            "compound_id": [2, 1],
            "visual_x": [0.1, 0.2],
            "visual_y": [0.3, 0.4],
        }
    )

    with pytest.raises(Exception, match="alignment"):
        _validate_coordinates(before, after)


def test_validate_coordinates_rejects_nonfinite_values():
    before = pd.DataFrame({"compound_id": [1, 2]})
    after = pd.DataFrame(
        {
            "compound_id": [1, 2],
            "visual_x": [0.1, np.nan],
            "visual_y": [0.3, 0.4],
        }
    )

    with pytest.raises(Exception, match="invalid"):
        _validate_coordinates(before, after)

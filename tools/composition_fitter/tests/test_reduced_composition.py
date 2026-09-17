from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from osnap.reduced_composition import replace_progenitor_reduced_composition


def test_replace_progenitor_reduced_composition_uses_direct_ye_and_canonical_network() -> None:
    profiles = pd.DataFrame(
        {
            "density": [1.0e5, 2.0e5],
            "temp": [1.0e9, 1.1e9],
            "r": [1.0e8, 2.0e8],
            "ye": [0.50, 0.55],
            "neut": [0.05, 0.10],
            "h1": [0.15, 0.10],
            "he4": [0.30, 0.40],
            "ti44": [0.10, 0.05],
            "fe56": [0.20, 0.15],
            "fe": [0.20, 0.20],
            "prot": [0.0, 0.0],
            "co56": [0.0, 0.0],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["neut", "h1", "he4", "ti44", "fe56", "fe"]}

    updated = replace_progenitor_reduced_composition(prog)
    network = updated["nuclear_network"]

    assert updated is not prog
    assert prog["nuclear_network"] == ["neut", "h1", "he4", "ti44", "fe56", "fe"]
    assert updated["original_nuclear_network"] == ["neut", "h1", "he4", "ti44", "fe56", "fe"]
    assert network == ["nt1", "h1", "he4", "ti44", "fe56"]
    assert "neut" not in updated["profiles"].columns
    assert "fe" not in updated["profiles"].columns
    assert "prot" not in updated["profiles"].columns
    assert "co56" not in updated["profiles"].columns
    assert np.allclose(updated["profiles"]["r"], profiles["r"])
    assert np.allclose(updated["profiles"][network].sum(axis=1), 1.0, atol=1.0e-10)

    z_over_a = np.array([0.0, 1.0, 0.5, 22.0 / 44.0, 26.0 / 56.0])
    assert np.allclose(updated["profiles"][network].to_numpy() @ z_over_a, profiles["ye"], atol=1.0e-10)
    assert np.allclose(updated["profiles"]["ti44"], 0.0, atol=1.0e-14)
    assert np.allclose(updated["reduced_composition_diagnostics"]["target_ye"], profiles["ye"])
    assert np.allclose(updated["reduced_composition_diagnostics"]["merged_fe_mass"], profiles["fe"])


def test_replace_progenitor_reduced_composition_requires_direct_ye() -> None:
    profiles = pd.DataFrame({"h1": [0.5], "he4": [0.5]})
    prog = {"profiles": profiles, "nuclear_network": ["h1", "he4"]}

    with pytest.raises(ValueError, match="direct Ye"):
        replace_progenitor_reduced_composition(prog)


def test_replace_progenitor_reduced_composition_supports_unstable_override() -> None:
    profiles = pd.DataFrame(
        {
            "ye": [0.5],
            "h1": [0.25],
            "he4": [0.50],
            "ni56": [0.25],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["h1", "he4", "ni56"]}

    updated = replace_progenitor_reduced_composition(prog, unstable_isotopes=("ti44",))

    assert updated["profiles"]["ni56"].iloc[0] > 0.0
    assert updated["profiles"][updated["nuclear_network"]].sum(axis=1).iloc[0] == pytest.approx(1.0)

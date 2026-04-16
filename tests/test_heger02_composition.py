from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from composition_fitter.fitter import Heger02CompositionFitter
from osnap.heger02_composition import replace_progenitor_composition


def _mock_fitter() -> Heger02CompositionFitter:
    isotopes = np.array(["nt1", "h1", "he4", "c12", "o16"])
    z_over_a = np.array([0.0, 1.0, 0.5, 0.5, 0.5], dtype=np.float32)
    compositions = np.array(
        [
            [0.2, 0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.2, 0.8, 0.0, 0.0],
            [0.0, 0.4, 0.6, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    feature_matrix = np.array(
        [
            [5.0, 9.0, 0.4],
            [5.2, 9.1, 0.5],
            [5.4, 9.2, 0.6],
            [5.6, 9.3, 0.7],
        ],
        dtype=np.float32,
    )
    artifact = {
        "compositions": compositions,
        "feature_matrix": feature_matrix,
        "feature_mean": feature_matrix.mean(axis=0).astype(np.float32),
        "feature_std": feature_matrix.std(axis=0).astype(np.float32),
        "feature_min": feature_matrix.min(axis=0).astype(np.float32),
        "feature_max": feature_matrix.max(axis=0).astype(np.float32),
        "ye": (compositions.astype(np.float64) @ z_over_a.astype(np.float64)).astype(np.float32),
        "z_over_a": z_over_a,
        "isotopes": isotopes,
    }
    return Heger02CompositionFitter(artifact, initial_k=4)


def test_replace_progenitor_composition_uses_direct_ye_and_fitter_basis() -> None:
    fitter = _mock_fitter()
    profiles = pd.DataFrame(
        {
            "density": [1.0e5, 2.0e5],
            "temp": [1.0e9, 1.1e9],
            "r": [1.0e8, 2.0e8],
            "ye": [0.5, 0.6],
            "neut": [0.2, 0.1],
            "he4": [0.8, 0.9],
            "prot": [0.0, 0.0],
            "co56": [0.0, 0.0],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["neut", "he4"]}

    updated = replace_progenitor_composition(prog, fitter=fitter)
    network = fitter.isotopes.tolist()

    assert updated is not prog
    assert prog["nuclear_network"] == ["neut", "he4"]
    assert updated["original_nuclear_network"] == ["neut", "he4"]
    assert updated["nuclear_network"] == network
    assert "neut" not in updated["profiles"].columns
    assert "prot" not in updated["profiles"].columns
    assert "co56" not in updated["profiles"].columns
    assert all(isotope in updated["profiles"].columns for isotope in network)
    assert np.allclose(updated["profiles"]["r"], profiles["r"])
    assert np.allclose(updated["profiles"][network].sum(axis=1), 1.0, atol=1.0e-6)
    assert np.allclose(updated["heger02_composition_diagnostics"]["requested_ye"], [0.5, 0.6])
    assert np.allclose(updated["heger02_composition_diagnostics"]["fitted_ye"], [0.5, 0.6], atol=1.0e-6)


def test_replace_progenitor_composition_derives_ye_from_aliases_and_fe_bucket() -> None:
    fitter = _mock_fitter()
    profiles = pd.DataFrame(
        {
            "density": [2.5e5],
            "temp": [1.15e9],
            "neut": [0.10],
            "prot": [0.20],
            "he4": [0.40],
            "fe": [0.30],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["neut", "prot", "he4", "fe"]}

    updated = replace_progenitor_composition(prog, fitter=fitter)
    expected_ye = 0.10 * 0.0 + 0.20 * 1.0 + 0.40 * 0.5 + 0.30 * (26.0 / 56.0)

    assert "neut" not in updated["profiles"].columns
    assert "prot" not in updated["profiles"].columns
    assert "fe" not in updated["profiles"].columns
    assert np.isclose(updated["heger02_composition_diagnostics"].loc[0, "requested_ye"], expected_ye)
    assert np.isclose(updated["heger02_composition_diagnostics"].loc[0, "fitted_ye"], expected_ye, atol=1.0e-6)


def test_replace_progenitor_composition_direct_ye_does_not_parse_source_network() -> None:
    fitter = _mock_fitter()
    profiles = pd.DataFrame(
        {
            "density": [3.0e5],
            "temp": [1.2e9],
            "ye": [0.6],
            "unknown_bucket": [1.0],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["unknown_bucket"]}

    updated = replace_progenitor_composition(prog, fitter=fitter)

    assert "unknown_bucket" not in updated["profiles"].columns
    assert np.isclose(updated["heger02_composition_diagnostics"].loc[0, "requested_ye"], 0.6)


def test_replace_progenitor_composition_preserves_clipping_diagnostics() -> None:
    fitter = _mock_fitter()
    profiles = pd.DataFrame(
        {
            "density": [1.0e5],
            "temp": [1.0e9],
            "ye": [0.95],
            "he4": [1.0],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["he4"]}

    updated = replace_progenitor_composition(prog, fitter=fitter)
    diagnostics = updated["heger02_composition_diagnostics"]

    assert bool(diagnostics.loc[0, "clipped"])
    assert np.isclose(diagnostics.loc[0, "requested_ye"], 0.95)
    assert np.isclose(diagnostics.loc[0, "query_ye"], 0.7, atol=1.0e-6)
    assert "Query clipped" in diagnostics.loc[0, "warnings"]


def test_replace_progenitor_composition_requires_parseable_composition_for_fallback_ye() -> None:
    fitter = _mock_fitter()
    profiles = pd.DataFrame(
        {
            "density": [1.0e5],
            "temp": [1.0e9],
            "unknown_bucket": [1.0],
        }
    )
    prog = {"profiles": profiles, "nuclear_network": ["unknown_bucket"]}

    with pytest.raises(ValueError, match="Unsupported isotope label"):
        replace_progenitor_composition(prog, fitter=fitter)

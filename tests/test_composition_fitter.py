from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from composition_fitter.fitter import Heger02CompositionFitter
from composition_fitter.heger02_dataset import build_artifact, build_reduced_artifact, load_artifact
from composition_fitter.reduced_network import ReducedNetwork, build_reduction_recipe, load_reduced_network, project_compositions
from composition_fitter.sukhbold_plot import (
    DEFAULT_PLOTTED_SELECTORS as COMPARISON_PLOTTED_SELECTORS,
    plot_sukhbold_profile_comparison,
)
from composition_fitter.sukhbold_profile import read_sukhbold_profile
from composition_fitter.sukhbold_two_panel_plot import (
    DEFAULT_PLOTTED_SELECTORS as TWO_PANEL_PLOTTED_SELECTORS,
    plot_sukhbold_two_panel_comparison,
)
from composition_fitter.sukhbold_ye_plot import (
    compute_composition_ye,
    plot_sukhbold_ye_comparison,
    temperature_threshold_mass,
)
from composition_fitter.viewer import prepare_plot_arrays, stabilize_query_to_limits

HEADER_OFFSET = 7
HEADER_WIDTH = 25


def _header_line(columns: list[str]) -> str:
    return (" " * HEADER_OFFSET) + "".join(f"{column:<{HEADER_WIDTH}}" for column in columns)


def _write_mock_presn(path: Path, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VERSION 10104 BURN -- mock\n")
        handle.write(_header_line(columns) + "\n")
        for row_number, row in enumerate(rows, start=1):
            handle.write(f"{row_number:4d}: " + " ".join(row) + "\n")


def _build_mock_source_dir(tmp_path: Path) -> Path:
    source_dir = tmp_path / "progenitors" / "heger02"
    source_dir.mkdir(parents=True)
    columns_a = [
        "cell outer total mass",
        "cell outer radius",
        "cell outer velocity",
        "cell density",
        "cell temperature",
        "cell pressure",
        "cell specific energy",
        "cell specific entropy",
        "cell angular velocity",
        "cell A_bar",
        "cell Y_e",
        "stability",
        "nt1",
        "h1",
        "he4",
        "c12",
    ]
    columns_b = [
        "cell outer total mass",
        "cell outer radius",
        "cell outer velocity",
        "cell density",
        "cell temperature",
        "cell pressure",
        "cell specific energy",
        "cell specific entropy",
        "cell angular velocity",
        "cell A_bar",
        "cell Y_e",
        "stability",
        "nt1",
        "h1",
        "he4",
        "o16",
    ]
    rows_a = [
        [
            "1.0e33",
            "1.0e8",
            "0.0",
            "1.0e5",
            "1.0e9",
            "1.0e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.75",
            "radiative",
            "0.00",
            "0.50",
            "0.50",
            "0.00",
        ],
        [
            "1.1e33",
            "1.1e8",
            "0.0",
            "2.0e5",
            "1.1e9",
            "1.1e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.50",
            "convective",
            "0.25",
            "0.25",
            "0.50",
            "0.00",
        ],
    ]
    rows_b = [
        [
            "1.2e33",
            "1.2e8",
            "0.0",
            "3.0e5",
            "1.2e9",
            "1.2e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.625",
            "radiative",
            "0.00",
            "0.25",
            "0.50",
            "0.25",
        ],
        [
            "1.3e33",
            "1.3e8",
            "0.0",
            "4.0e5",
            "1.3e9",
            "1.3e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.55",
            "overshooting",
            "0.10",
            "0.20",
            "0.40",
            "0.30",
        ],
    ]

    _write_mock_presn(source_dir / "mockA@presn", columns_a, rows_a)
    _write_mock_presn(source_dir / "mockB@presn", columns_b, rows_b)
    return source_dir


def _build_mock_dataset(tmp_path: Path) -> Path:
    source_dir = _build_mock_source_dir(tmp_path)
    artifact_path = tmp_path / "output" / "mock_artifact.npz"
    build_artifact(source_dir, artifact_path)
    return artifact_path


def _write_mock_sukhbold(path: Path, labels: list[str], rows: list[list[str]]) -> None:
    header = (
        "  grid    cell outer total mass        cell outer radius      cell outer velocity"
        "             cell density         cell temperature            cell pressure"
        "     cell specific energy    cell specific entropy    cell angular velocity"
        "               cell A_bar                 cell Y_e       stability   NETWORK                 "
        + " ".join(labels)
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("VERSION 10111 -- mock sukhbold\n")
        handle.write(header + "\n")
        for row_number, row in enumerate(rows, start=1):
            handle.write(f"{row_number}: " + " ".join(row) + "\n")


def _build_mock_sukhbold_profile(tmp_path: Path) -> Path:
    profile_path = tmp_path / "progenitors" / "sukhbold_2016" / "s25.0_presn"
    profile_path.parent.mkdir(parents=True)
    labels = ["neutrons", "H1", "He4", "C12", "O16"]
    rows = [
        [
            "1.0e33",
            "1.0e8",
            "0.0",
            "1.0e5",
            "1.0e9",
            "1.0e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.60",
            "radiative",
            "NSE",
            "0.10",
            "0.20",
            "0.40",
            "---",
            "0.30",
        ],
        [
            "1.1e33",
            "1.5e8",
            "0.0",
            "3.0e5",
            "1.2e9",
            "1.2e20",
            "1.0e17",
            "1.0",
            "0.0",
            "2.0",
            "0.55",
            "convective",
            "APPROX",
            "0.15",
            "0.15",
            "0.35",
            "0.05",
            "0.30",
        ],
    ]
    _write_mock_sukhbold(profile_path, labels, rows)
    return profile_path


def _write_reduced_net(path: Path, labels: list[str]) -> None:
    path.write_text(", ".join(f"'{label}'" for label in labels), encoding="utf-8")


def _write_sunet(path: Path, labels: list[str]) -> None:
    path.write_text("\n".join(labels) + "\n", encoding="utf-8")


def test_build_artifact_aligns_union_basis(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    artifact = load_artifact(artifact_path)

    isotopes = artifact["isotopes"].tolist()
    assert isotopes == ["nt1", "h1", "he4", "c12", "o16"]
    assert artifact["compositions"].shape == (4, 5)
    assert np.allclose(artifact["compositions"].sum(axis=1), 1.0)
    assert np.all(artifact["compositions"] >= 0.0)
    assert artifact["feature_matrix"].shape == (4, 3)
    assert artifact["artifact_mode"][0] == "full"


def test_predict_enforces_constraints_and_quantiles(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)

    prediction = fitter.predict(2.5e5, 1.15e9, 0.60)
    assert np.all(prediction.mass_fractions >= 0.0)
    assert np.isclose(prediction.mass_fractions.sum(), 1.0, atol=1e-6)

    artifact = load_artifact(artifact_path)
    z_over_a = artifact["z_over_a"].astype(np.float64)
    ye = float(prediction.mass_fractions.astype(np.float64) @ z_over_a)
    assert np.isclose(ye, 0.60, atol=1e-6)
    assert np.all(prediction.q16 <= prediction.q50)
    assert np.all(prediction.q50 <= prediction.q84)


def test_prepare_plot_arrays_is_nonnegative(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)

    prediction = fitter.predict(2.0e5, 1.0e9, 0.40, nuclei=["h1", "he4"])
    labels, values, yerr = prepare_plot_arrays(prediction)
    assert labels == ["h1", "he4"]
    assert values.shape == (2,)
    assert yerr.shape == (2, 2)
    assert np.all(yerr >= 0.0)


def test_predict_supports_grouped_element_selectors(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)

    prediction = fitter.predict(3.5e5, 1.25e9, 0.58, nuclei=["he4", "o16", "c"])
    labels = prediction.isotopes.tolist()
    assert labels == ["he4", "o16", "c"]
    assert prediction.mass_fractions.shape == (3,)
    assert np.all(prediction.mass_fractions >= 0.0)


def test_predict_clips_to_local_ye_band_when_neighborhood_is_too_small(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=2, max_k=2)

    prediction = fitter.predict(2.5e5, 1.15e9, 0.70)
    assert prediction.warnings
    assert any("Local Ye clipped" in warning for warning in prediction.warnings)
    artifact = load_artifact(artifact_path)
    ye = float(prediction.mass_fractions.astype(np.float64) @ artifact["z_over_a"].astype(np.float64))
    assert np.isclose(ye, prediction.query[2], atol=1e-6)


def test_local_valid_ranges_and_stabilization(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4, limit_k=2)

    query = np.array([5.50, 9.05, 0.60], dtype=np.float64)
    limits = fitter.local_valid_ranges(query)
    assert limits.shape == (3, 2)
    assert np.all(limits[:, 0] <= limits[:, 1])

    pushed = np.array([10.0, 10.0, 0.70], dtype=np.float64)
    stabilized, stabilized_limits = stabilize_query_to_limits(fitter, pushed)
    assert np.all(stabilized >= stabilized_limits[:, 0] - 1.0e-12)
    assert np.all(stabilized <= stabilized_limits[:, 1] + 1.0e-12)


def test_read_sukhbold_profile_normalizes_labels(tmp_path: Path) -> None:
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    profile = read_sukhbold_profile(profile_path)

    assert profile.source_labels.tolist() == ["neutrons", "H1", "He4", "C12", "O16"]
    assert profile.selector_labels.tolist() == ["nt1", "h1", "he4", "c12", "o16"]
    assert profile.source_abundances.shape == (2, 5)
    assert np.isclose(profile.source_abundances[0, 3], 0.0)
    assert np.allclose(profile.enclosed_mass_g, [1.0e33, 1.1e33])
    assert np.allclose(profile.log_radius_cm, np.log10(profile.radius_cm))


def test_predict_profile_returns_full_and_projected_outputs(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)
    profile = read_sukhbold_profile(profile_path)

    result = fitter.predict_profile(profile)
    artifact = load_artifact(artifact_path)

    assert result.predicted_isotopes.tolist() == artifact["isotopes"].tolist()
    assert result.predicted_mass_fractions.shape == (2, 5)
    assert result.predicted_q16.shape == (2, 5)
    assert result.projected_mass_fractions.shape == (2, 5)
    assert result.projected_q84.shape == (2, 5)
    assert result.source_labels.tolist() == profile.source_labels.tolist()
    assert np.allclose(result.enclosed_mass_g, profile.enclosed_mass_g)
    assert result.predicted_ye.shape == (2,)

    z_over_a = artifact["z_over_a"].astype(np.float64)
    for zone_index in range(result.predicted_mass_fractions.shape[0]):
        zone_ye = float(result.predicted_mass_fractions[zone_index].astype(np.float64) @ z_over_a)
        assert np.isclose(zone_ye, float(profile.ye[zone_index]), atol=1e-6)
        assert np.isclose(zone_ye, float(result.predicted_ye[zone_index]), atol=1e-6)


def test_plot_sukhbold_profile_comparison_creates_lines_and_bands(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)
    result = fitter.predict_profile(read_sukhbold_profile(profile_path))

    figure = plot_sukhbold_profile_comparison(result)
    axis = figure.axes[0]
    ye_axis = figure.axes[1]
    plotted_count = int(np.isin(result.selector_labels, COMPARISON_PLOTTED_SELECTORS).sum())

    assert len(axis.lines) == 2 * plotted_count
    assert len(axis.collections) >= plotted_count
    assert axis.get_xlabel() == "Enclosed mass (M_sun)"
    assert ye_axis.get_ylabel() == "Electron fraction Ye"
    assert len(ye_axis.lines) == 2
    plt.close(figure)


def test_plot_sukhbold_two_panel_comparison_creates_panel_lines(tmp_path: Path) -> None:
    artifact_path = _build_mock_dataset(tmp_path)
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)
    result = fitter.predict_profile(read_sukhbold_profile(profile_path))

    figure = plot_sukhbold_two_panel_comparison(result)
    top_axis, bottom_axis = figure.axes

    plotted_count = int(np.isin(result.selector_labels, TWO_PANEL_PLOTTED_SELECTORS).sum())
    assert len(top_axis.lines) == plotted_count
    assert len(bottom_axis.lines) == plotted_count
    assert bottom_axis.get_xlabel() == "Enclosed mass (M_sun)"
    assert top_axis.get_ylabel() == "Mass fraction"
    assert bottom_axis.get_ylabel() == "Mass fraction"
    plt.close(figure)


def test_compute_composition_ye_matches_expected_weights(tmp_path: Path) -> None:
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    profile = read_sukhbold_profile(profile_path)

    ye = compute_composition_ye(profile)
    expected = np.array(
        [
            0.10 * 0.0 + 0.20 * 1.0 + 0.40 * 0.5 + 0.00 * 0.5 + 0.30 * 0.5,
            0.15 * 0.0 + 0.15 * 1.0 + 0.35 * 0.5 + 0.05 * 0.5 + 0.30 * 0.5,
        ],
        dtype=np.float64,
    )
    assert np.allclose(ye, expected)


def test_temperature_threshold_mass_returns_outermost_hot_coordinate(tmp_path: Path) -> None:
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    profile = read_sukhbold_profile(profile_path)

    threshold_mass = temperature_threshold_mass(profile, temperature_threshold=1.1e9)
    assert np.isclose(threshold_mass, 1.1e33 / 1.9884098706980504e33)

    assert temperature_threshold_mass(profile, temperature_threshold=6.0e9) is None


def test_plot_sukhbold_ye_comparison_creates_expected_lines(tmp_path: Path) -> None:
    profile_path = _build_mock_sukhbold_profile(tmp_path)
    profile = read_sukhbold_profile(profile_path)

    figure = plot_sukhbold_ye_comparison(
        [profile, profile],
        labels=["model-a", "model-b"],
        temperature_threshold=1.1e9,
    )
    axis = figure.axes[0]

    assert len(axis.lines) == 6
    assert axis.get_xlabel() == "Enclosed mass (M_sun)"
    assert axis.get_ylabel() == "Electron fraction Ye"
    plt.close(figure)


def test_load_reduced_network_normalizes_neut_alias(tmp_path: Path) -> None:
    reduced_net_path = tmp_path / "reducedNet.txt"
    _write_reduced_net(reduced_net_path, ["neut", "h1", "he4", "c12"])

    reduced_net = load_reduced_network(reduced_net_path)
    assert reduced_net.isotopes.tolist() == ["nt1", "h1", "he4", "c12"]


def test_load_reduced_network_parses_sunet_aliases(tmp_path: Path) -> None:
    reduced_net_path = tmp_path / "mock.sunet"
    _write_sunet(reduced_net_path, ["n", "p", "d", "t", "he4", "c12"])

    reduced_net = load_reduced_network(reduced_net_path)
    assert reduced_net.isotopes.tolist() == ["nt1", "h1", "h2", "h3", "he4", "c12"]


def test_reduction_recipe_interpolates_within_same_element_chain() -> None:
    reduced_net = ReducedNetwork(
        path=Path("dummy"),
        isotopes=np.array(["nt1", "h1", "c12", "c14"]),
        a=np.array([1, 1, 12, 14], dtype=np.int16),
        z=np.array([0, 1, 6, 6], dtype=np.int16),
        z_over_a=np.array([0.0, 1.0, 0.5, 6.0 / 14.0], dtype=np.float32),
    )
    recipe = build_reduction_recipe(["c13"], reduced_net)
    reduced, qa = project_compositions(
        np.array([[1.0]], dtype=np.float64),
        np.array([6.0 / 13.0], dtype=np.float64),
        recipe,
        reduced_net.z_over_a.astype(np.float64),
    )

    assert np.isclose(reduced.sum(), 1.0, atol=1e-6)
    assert np.isclose(reduced[0, 0], 0.0)
    assert np.isclose(reduced[0, 1], 0.0)
    assert np.isclose(reduced[0] @ reduced_net.z_over_a.astype(np.float64), 6.0 / 13.0, atol=1e-6)
    assert np.isclose(qa["light_correction_mass"][0], 0.0, atol=1e-7)


def test_reduction_recipe_maps_missing_elements_to_light_species() -> None:
    reduced_net = ReducedNetwork(
        path=Path("dummy"),
        isotopes=np.array(["nt1", "h1"]),
        a=np.array([1, 1], dtype=np.int16),
        z=np.array([0, 1], dtype=np.int16),
        z_over_a=np.array([0.0, 1.0], dtype=np.float32),
    )
    recipe = build_reduction_recipe(["o16"], reduced_net)
    reduced, qa = project_compositions(
        np.array([[1.0]], dtype=np.float64),
        np.array([0.5], dtype=np.float64),
        recipe,
        reduced_net.z_over_a.astype(np.float64),
    )

    assert np.allclose(reduced[0], [0.5, 0.5], atol=1e-6)
    assert np.isclose(qa["missing_element_mass"][0], 1.0, atol=1e-6)


def test_build_reduced_artifact_preserves_ye_and_is_queryable(tmp_path: Path) -> None:
    source_dir = _build_mock_source_dir(tmp_path)
    reduced_net_path = tmp_path / "mock.sunet"
    _write_sunet(reduced_net_path, ["n", "p", "he4", "c12", "o16", "li6"])

    artifact_path = tmp_path / "output" / "mock_reduced_artifact.npz"
    build_reduced_artifact(source_dir, artifact_path, reduced_net_path)
    artifact = load_artifact(artifact_path)
    fitter = Heger02CompositionFitter.load(artifact_path, initial_k=4)

    assert artifact["artifact_mode"][0] == "reduced"
    assert artifact["isotopes"].tolist() == ["nt1", "h1", "he4", "c12", "o16", "li6"]
    assert artifact["target_net_path"][0] == str(reduced_net_path)
    assert np.allclose(artifact["compositions"].sum(axis=1), 1.0, atol=1e-6)
    assert np.max(np.abs(artifact["projection_ye_error_after"])) <= 1.0e-6
    assert np.allclose(artifact["compositions"][:, -1], 0.0)

    prediction = fitter.predict(2.5e5, 1.15e9, 0.60)
    ye = float(prediction.mass_fractions.astype(np.float64) @ artifact["z_over_a"].astype(np.float64))
    assert np.isclose(ye, 0.60, atol=1e-6)

"""Validation plots for Ye-repaired reduced Sukhbold compositions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .reduced_composition import (
    DEFAULT_UNSTABLE_ISOTOPES,
    RepairedSukhboldComposition,
    repair_sukhbold_profile,
)
from .sukhbold_profile import read_sukhbold_profile
from .sukhbold_ye_plot import DEFAULT_SUKHBOLD_YE_PROFILE_DIR, M_SUN_G

DEFAULT_PLOTTED_SELECTORS = ("fe56", "ni56", "fe54", "cr48", "ti44", "si28", "o16", "c12")
DEFAULT_OUTPUT_DIR = Path("output/reduced_composition_repair")
DEFAULT_REPAIR_PROFILES = (
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s12.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s15.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s20.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s23.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s24.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s25.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s33_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s35_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s100_presn",
)


def plot_repaired_sukhbold_ye_comparison(
    results: Sequence[RepairedSukhboldComposition],
    *,
    labels: Sequence[str] | None = None,
    x_limits: tuple[float, float] = (1.0, 4.0),
):
    """Plot direct, original-composition, and repaired-composition Ye."""

    if not results:
        raise ValueError("At least one repaired Sukhbold composition is required.")
    if labels is None:
        labels = [result.path.name.replace("_presn", "") for result in results]
    elif len(labels) != len(results):
        raise ValueError("labels must match the number of results.")

    figure, (ye_ax, residual_ax) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    cmap = plt.get_cmap("tab10", len(results))
    all_ye_in_range: list[np.ndarray] = []

    for index, (result, label) in enumerate(zip(results, labels)):
        color = cmap(index)
        x_values = result.enclosed_mass_g.astype(np.float64) / M_SUN_G
        column_ye = result.target_ye.astype(np.float64)
        original_ye = result.diagnostics["original_composition_ye"].astype(np.float64)
        repaired_ye = result.diagnostics["repaired_composition_ye"].astype(np.float64)

        ye_ax.plot(x_values, column_ye, color=color, linewidth=1.8)
        ye_ax.plot(x_values, original_ye, color=color, linewidth=1.3, linestyle=":")
        ye_ax.plot(x_values, repaired_ye, color=color, linewidth=1.4, linestyle="--")
        residual_ax.plot(x_values, original_ye - column_ye, color=color, linewidth=1.2, linestyle=":")
        residual_ax.plot(x_values, repaired_ye - column_ye, color=color, linewidth=1.4)

        in_range = (x_values >= x_limits[0]) & (x_values <= x_limits[1])
        if np.any(in_range):
            all_ye_in_range.extend((column_ye[in_range], original_ye[in_range], repaired_ye[in_range]))

    ye_ax.set_ylabel("Electron fraction Ye")
    ye_ax.set_title("Reduced Sukhbold composition Ye repair")
    ye_ax.grid(True, alpha=0.25)
    if all_ye_in_range:
        y_values = np.concatenate(all_ye_in_range)
        y_span = float(np.nanmax(y_values) - np.nanmin(y_values))
        padding = max(0.005, 0.08 * y_span)
        ye_ax.set_ylim(max(0.0, float(np.nanmin(y_values) - padding)), min(1.0, float(np.nanmax(y_values) + padding)))

    residual_ax.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.8)
    residual_ax.set_xlabel("Enclosed mass (M_sun)")
    residual_ax.set_ylabel("Ye residual")
    residual_ax.set_xlim(*x_limits)
    residual_ax.set_yscale("symlog", linthresh=1.0e-10, linscale=0.8)
    residual_ax.grid(True, which="both", alpha=0.25)

    model_handles = [Line2D([], [], color=cmap(index), linewidth=2.0, label=label) for index, label in enumerate(labels)]
    style_handles = [
        Line2D([], [], color="black", linewidth=1.8, linestyle="-", label="column Ye"),
        Line2D([], [], color="black", linewidth=1.3, linestyle=":", label="original composition Ye"),
        Line2D([], [], color="black", linewidth=1.4, linestyle="--", label="repaired composition Ye"),
    ]
    legend = ye_ax.legend(handles=model_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model")
    ye_ax.add_artist(legend)
    ye_ax.legend(handles=style_handles, loc="upper right", title="Style")
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return figure


def plot_repaired_sukhbold_abundance_comparison(
    result: RepairedSukhboldComposition,
    *,
    selectors: Sequence[str] = DEFAULT_PLOTTED_SELECTORS,
    y_floor: float = 1.0e-30,
):
    """Plot original and repaired abundances plus their signed difference."""

    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive for log-scale plotting.")

    label_to_index = {str(label): index for index, label in enumerate(result.labels.tolist())}
    selected = [(selector, label_to_index[selector]) for selector in selectors if selector in label_to_index]
    if not selected:
        raise ValueError("None of the requested isotopes are present in the repaired composition.")

    figure, (abundance_ax, delta_ax) = plt.subplots(
        2,
        1,
        figsize=(15, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    x_values = result.enclosed_mass_g.astype(np.float64) / M_SUN_G
    cmap = plt.get_cmap("tab20", len(selected))

    for color_index, (label, column_index) in enumerate(selected):
        color = cmap(color_index)
        original = result.original_mass_fractions[:, column_index].astype(np.float64)
        repaired = result.mass_fractions[:, column_index].astype(np.float64)
        abundance_ax.plot(x_values, np.clip(original, y_floor, None), color=color, linewidth=1.25, label=label)
        abundance_ax.plot(x_values, np.clip(repaired, y_floor, None), color=color, linewidth=1.15, linestyle="--")
        delta_ax.plot(x_values, repaired - original, color=color, linewidth=1.15)

    abundance_ax.set_ylabel("Mass fraction")
    abundance_ax.set_yscale("log")
    abundance_ax.set_title(f"Original and Ye-repaired reduced composition: {result.path.name}")
    abundance_ax.grid(True, which="both", alpha=0.25)

    delta_ax.axhline(0.0, color="0.25", linewidth=0.9, alpha=0.8)
    delta_ax.set_xlabel("Enclosed mass (M_sun)")
    delta_ax.set_ylabel("X_repaired - X_original")
    delta_ax.set_yscale("symlog", linthresh=1.0e-10, linscale=0.8)
    delta_ax.grid(True, which="both", alpha=0.25)

    species_legend = abundance_ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.52),
        borderaxespad=0.0,
        title="Species",
        fontsize=8,
    )
    abundance_ax.add_artist(species_legend)
    abundance_ax.legend(
        handles=[
            Line2D([0], [0], color="0.25", linewidth=1.4, linestyle="-", label="original"),
            Line2D([0], [0], color="0.25", linewidth=1.4, linestyle="--", label="repaired"),
        ],
        loc="lower left",
        fontsize=8,
    )
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return figure


def plot_repaired_sukhbold_diagnostics(result: RepairedSukhboldComposition):
    """Plot the main repair diagnostics against enclosed mass."""

    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    x_values = result.enclosed_mass_g.astype(np.float64) / M_SUN_G
    diagnostics = result.diagnostics

    axes[0].plot(x_values, diagnostics["zeroed_unstable_mass"], color="tab:red", linewidth=1.4)
    axes[0].set_ylabel("Zeroed mass")
    axes[0].set_title(f"Reduced-composition repair diagnostics: {result.path.name}")

    axes[1].plot(x_values, diagnostics["projection_l2_delta"], color="tab:blue", linewidth=1.4)
    axes[1].set_ylabel("Projection L2")

    ye_abs_error = np.abs(diagnostics["ye_error"])
    axes[2].plot(x_values, ye_abs_error, color="tab:green", linewidth=1.4)
    axes[2].set_xlabel("Enclosed mass (M_sun)")
    axes[2].set_ylabel("|Ye residual|")
    if np.any(ye_abs_error > 0.0):
        axes[2].set_yscale("log")
    else:
        axes[2].set_ylim(0.0, 1.0e-12)

    for axis in axes:
        axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot validation figures for reduced-composition Ye repair.")
    parser.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        default=list(DEFAULT_REPAIR_PROFILES),
        help="One or more Sukhbold profile paths to repair and plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for generated validation plots.",
    )
    parser.add_argument(
        "--unstable-isotopes",
        nargs="*",
        default=list(DEFAULT_UNSTABLE_ISOTOPES),
        help="Reduced-network isotope labels to zero before Ye repair.",
    )
    parser.add_argument("--x-min", type=float, default=1.0, help="Minimum enclosed mass in solar masses for Ye plot.")
    parser.add_argument("--x-max", type=float, default=4.0, help="Maximum enclosed mass in solar masses for Ye plot.")
    args = parser.parse_args()

    profiles = [read_sukhbold_profile(path) for path in args.profiles]
    results = [repair_sukhbold_profile(profile, unstable_isotopes=args.unstable_isotopes) for profile in profiles]
    labels = [profile.path.name.replace("_presn", "") for profile in profiles]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ye_figure = plot_repaired_sukhbold_ye_comparison(results, labels=labels, x_limits=(args.x_min, args.x_max))
    ye_figure.savefig(args.output_dir / "reduced_composition_ye_repair.png", dpi=200)
    plt.close(ye_figure)

    for result in results:
        stem = result.path.name.replace("_presn", "")
        abundance_figure = plot_repaired_sukhbold_abundance_comparison(result)
        abundance_figure.savefig(args.output_dir / f"{stem}_abundance_repair.png", dpi=200)
        plt.close(abundance_figure)

        diagnostics_figure = plot_repaired_sukhbold_diagnostics(result)
        diagnostics_figure.savefig(args.output_dir / f"{stem}_repair_diagnostics.png", dpi=200)
        plt.close(diagnostics_figure)


if __name__ == "__main__":
    main()

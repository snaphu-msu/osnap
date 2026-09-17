"""Plot direct and composition-derived electron fractions for Sukhbold profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from .isotopes import parse_isotope
from .sukhbold_profile import SukhboldProfile, read_sukhbold_profile

M_SUN_G = 1.9884098706980504e33
DEFAULT_SUKHBOLD_YE_PROFILE_DIR = Path("progenitors/sukhbold_2016")
DEFAULT_SUKHBOLD_YE_PROFILES = (
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s15.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s20.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s23.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s24.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s25.0_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s33_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s35_presn",
    DEFAULT_SUKHBOLD_YE_PROFILE_DIR / "s100_presn",
)


def composition_ye_weights(selector_labels: Sequence[str]) -> np.ndarray:
    """Return per-species ``Z/A`` weights for the reduced Sukhbold network."""

    weights = np.empty(len(selector_labels), dtype=np.float64)
    for index, label in enumerate(selector_labels):
        if label == "fe":
            # The final Sukhbold bucket is an element-like iron group label, not a true isotope.
            weights[index] = 26.0 / 56.0
        else:
            weights[index] = parse_isotope(str(label)).z_over_a
    return weights


def compute_composition_ye(profile: SukhboldProfile) -> np.ndarray:
    """Compute ``Ye`` from the Sukhbold abundance columns."""

    weights = composition_ye_weights(profile.selector_labels.tolist())
    return profile.source_abundances.astype(np.float64) @ weights


def temperature_threshold_mass(
    profile: SukhboldProfile,
    *,
    temperature_threshold: float = 6.0e9,
) -> float | None:
    """Return the outermost enclosed-mass coordinate where ``T`` exceeds a threshold."""

    hot_indices = np.flatnonzero(profile.temperature.astype(np.float64) > temperature_threshold)
    if hot_indices.size == 0:
        return None
    return float(profile.enclosed_mass_g[int(hot_indices[-1])] / M_SUN_G)


def plot_sukhbold_ye_comparison(
    profiles: Sequence[SukhboldProfile],
    *,
    labels: Sequence[str] | None = None,
    x_limits: tuple[float, float] = (0.25, 4.0),
    temperature_threshold: float = 6.0e9,
    ax=None,
):
    """Plot file-column and composition-derived ``Ye`` for one or more Sukhbold profiles."""

    if not profiles:
        raise ValueError("At least one Sukhbold profile is required.")

    if labels is None:
        labels = [profile.path.name.replace("_presn", "") for profile in profiles]
    elif len(labels) != len(profiles):
        raise ValueError("labels must match the number of profiles.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(11, 7))
    else:
        figure = ax.figure

    cmap = plt.get_cmap("tab10", len(profiles))
    all_y_in_range: list[np.ndarray] = []

    for index, (profile, label) in enumerate(zip(profiles, labels)):
        x_values = profile.enclosed_mass_g.astype(np.float64) / M_SUN_G
        column_ye = profile.ye.astype(np.float64)
        composition_ye = compute_composition_ye(profile)
        color = cmap(index)

        ax.plot(x_values, column_ye, color=color, linewidth=1.8)
        ax.plot(x_values, composition_ye, color=color, linewidth=1.5, linestyle="--")

        threshold_mass = temperature_threshold_mass(profile, temperature_threshold=temperature_threshold)
        if threshold_mass is not None:
            ax.axvline(threshold_mass, color="black", linewidth=1.0, linestyle="--", alpha=0.6, zorder=0)

        in_range = (x_values >= x_limits[0]) & (x_values <= x_limits[1])
        if np.any(in_range):
            all_y_in_range.append(column_ye[in_range])
            all_y_in_range.append(composition_ye[in_range])

    ax.set_xlabel("Enclosed mass (M_sun)")
    ax.set_ylabel("Electron fraction Ye")
    ax.set_title("Sukhbold models: file-column Ye vs composition-derived Ye")
    ax.set_xlim(*x_limits)
    ax.grid(True, alpha=0.25)

    if all_y_in_range:
        y_values = np.concatenate(all_y_in_range)
        y_span = float(y_values.max() - y_values.min())
        padding = max(0.005, 0.08 * y_span)
        ax.set_ylim(max(0.0, float(y_values.min() - padding)), min(1.0, float(y_values.max() + padding)))

    model_handles = [
        Line2D([], [], color=cmap(index), linewidth=2.0, label=label)
        for index, label in enumerate(labels)
    ]
    style_handles = [
        Line2D([], [], color="black", linewidth=1.8, linestyle="-", label="column Ye"),
        Line2D([], [], color="black", linewidth=1.5, linestyle="--", label="composition Ye"),
        Line2D([], [], color="black", linewidth=1.0, linestyle="--", alpha=0.6, label=f"T > {temperature_threshold:.1e} K"),
    ]
    legend = ax.legend(handles=model_handles, loc="center left", bbox_to_anchor=(1.02, 0.5), title="Model")
    ax.add_artist(legend)
    ax.legend(handles=style_handles, loc="upper right", title="Style")
    figure.text(0.02, 0.02, "Composition Ye uses Z/A from listed species; grouped Fe bucket uses 26/56.", fontsize=9)
    figure.tight_layout(rect=(0.0, 0.04, 0.82, 1.0))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot direct and composition-derived Ye for Sukhbold profiles.")
    parser.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        default=list(DEFAULT_SUKHBOLD_YE_PROFILES),
        help="One or more Sukhbold profile paths to plot.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to save the plot instead of showing it interactively.",
    )
    parser.add_argument(
        "--x-min",
        type=float,
        default=1.0,
        help="Minimum enclosed mass in solar masses.",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=4.0,
        help="Maximum enclosed mass in solar masses.",
    )
    args = parser.parse_args()

    profiles = [read_sukhbold_profile(path) for path in args.profiles]
    labels = [path.name.replace("_presn", "") for path in args.profiles]
    figure = plot_sukhbold_ye_comparison(profiles, labels=labels, x_limits=(args.x_min, args.x_max))

    if args.output is None:
        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

"""Plot Sukhbold profile abundances against Heger02-fitter predictions."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import numpy as np

from .fitter import Heger02CompositionFitter, ProfilePrediction
from .heger02_dataset import DEFAULT_ARTIFACT_PATH, DEFAULT_SOURCE_DIR, build_reduced_artifact
from .sukhbold_profile import DEFAULT_SUKHBOLD_PROFILE_PATH, read_sukhbold_profile

DEFAULT_PLOTTED_SELECTORS = ("h1", "he4", "c12", "o16", "si28", "ti44", "ni56", "fe56")
M_SUN_G = 1.9884098706980504e33


def ensure_artifact(artifact_path: Path, source_dir: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    return build_reduced_artifact(source_dir, artifact_path)


def plot_sukhbold_profile_comparison(
    result: ProfilePrediction,
    *,
    ax=None,
    y_floor: float = 1.0e-30,
):
    """Plot all Sukhbold network species versus enclosed mass on one combined axes."""

    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive for log-scale plotting.")

    if ax is None:
        figure, ax = plt.subplots(figsize=(15, 10))
    else:
        figure = ax.figure

    x_values = result.enclosed_mass_g.astype(np.float64) / M_SUN_G
    ye_axis = ax.twinx()
    selected_indices = [
        index for index, label in enumerate(result.selector_labels.tolist()) if label in DEFAULT_PLOTTED_SELECTORS
    ]
    if not selected_indices:
        raise ValueError("None of the requested default isotopes are present in the profile result.")

    species_count = len(selected_indices)
    cmap = plt.get_cmap("tab20", species_count)

    for color_index, index in enumerate(selected_indices):
        color = cmap(color_index)
        label = result.source_labels[index]
        display_label = label.replace("'", "")
        source = np.clip(result.source_abundances[:, index].astype(np.float64), y_floor, None)
        predicted = np.clip(result.projected_mass_fractions[:, index].astype(np.float64), y_floor, None)
        lower = np.clip(result.projected_q16[:, index].astype(np.float64), y_floor, None)
        upper = np.clip(result.projected_q84[:, index].astype(np.float64), y_floor, None)

        ax.fill_between(x_values, lower, upper, color=color, alpha=0.12, linewidth=0.0)
        ax.plot(x_values, source, color=color, linewidth=1.5, label=display_label)
        ax.plot(x_values, predicted, color=color, linewidth=1.2, linestyle="--")

    ax.set_xlabel("Enclosed mass (M_sun)")
    ax.set_ylabel("Mass fraction")
    ax.set_yscale("log")
    ax.set_title("Sukhbold s25.0 abundances vs Heger02-fitter predictions")
    ax.grid(True, which="both", alpha=0.25)

    sukhbold_ye = result.ye.astype(np.float64)
    predicted_ye = result.predicted_ye.astype(np.float64)
    ye_axis.plot(x_values, sukhbold_ye, color="black", linewidth=1.5, alpha=0.85)
    ye_axis.plot(x_values, predicted_ye, color="black", linewidth=1.2, linestyle="--", alpha=0.85)
    ye_axis.set_ylabel("Electron fraction Ye")
    ye_axis.set_ylim(
        max(0.0, float(min(sukhbold_ye.min(), predicted_ye.min()) - 0.02)),
        min(1.0, float(max(sukhbold_ye.max(), predicted_ye.max()) + 0.02)),
    )

    species_legend = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        title="Sukhbold species",
        fontsize=8,
    )
    ax.add_artist(species_legend)

    style_handles = [
        Line2D([], [], color="black", linewidth=1.5, linestyle="-", label="source"),
        Line2D([], [], color="black", linewidth=1.2, linestyle="--", label="predicted"),
        mpatches.Patch(color="black", alpha=0.12, label="q16-q84"),
        Line2D([], [], color="black", linewidth=1.5, linestyle="-", alpha=0.85, label="source Ye"),
        Line2D([], [], color="black", linewidth=1.2, linestyle="--", alpha=0.85, label="predicted Ye"),
    ]
    ax.legend(handles=style_handles, loc="upper right", title="Styles")

    clipped_count = int(np.count_nonzero(result.clipped))
    if clipped_count:
        figure.text(0.02, 0.02, f"Clipped zones: {clipped_count}", fontsize=9)

    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Sukhbold profile abundances against fitter predictions.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Cached Heger02 artifact path.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the raw Heger02 '*@presn' files if the artifact must be built.",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=DEFAULT_SUKHBOLD_PROFILE_PATH,
        help="Sukhbold profile path to compare against.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional file path to save the plot instead of showing it interactively.",
    )
    parser.add_argument(
        "--y-floor",
        type=float,
        default=1.0e-30,
        help="Display-only lower floor for the log-scale mass fractions.",
    )
    args = parser.parse_args()

    artifact_path = ensure_artifact(args.artifact, args.source)
    fitter = Heger02CompositionFitter.load(artifact_path)
    profile = read_sukhbold_profile(args.profile)
    result = fitter.predict_profile(profile)
    figure = plot_sukhbold_profile_comparison(result, y_floor=args.y_floor)

    if args.output is None:
        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

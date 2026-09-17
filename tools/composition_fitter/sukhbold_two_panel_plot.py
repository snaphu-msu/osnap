"""Two-panel Sukhbold plot: source abundances above, fitted abundances below."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .fitter import Heger02CompositionFitter, ProfilePrediction
from .heger02_dataset import DEFAULT_ARTIFACT_PATH, DEFAULT_SOURCE_DIR, build_reduced_artifact
from .sukhbold_profile import DEFAULT_SUKHBOLD_PROFILE_PATH, read_sukhbold_profile

M_SUN_G = 1.9884098706980504e33
DEFAULT_PLOTTED_SELECTORS = ("fe56", "ni56", "fe54", "cr48", "ti44", "si28", "o16", "c12")


def ensure_artifact(artifact_path: Path, source_dir: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    return build_reduced_artifact(source_dir, artifact_path)


def plot_sukhbold_two_panel_comparison(
    result: ProfilePrediction,
    *,
    y_floor: float = 1.0e-30,
):
    """Plot source Sukhbold abundances above predicted projected abundances."""

    if y_floor <= 0.0:
        raise ValueError("y_floor must be positive for log-scale plotting.")

    figure, (source_ax, predicted_ax) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    x_values = result.enclosed_mass_g.astype(np.float64) / M_SUN_G
    selected_indices = [
        index
        for selector in DEFAULT_PLOTTED_SELECTORS
        for index, label in enumerate(result.selector_labels.tolist())
        if label == selector
    ]
    if not selected_indices:
        raise ValueError("None of the requested isotopes are present in the profile result.")

    species_count = len(selected_indices)
    cmap = plt.get_cmap("tab20", species_count)

    for color_index, source_index in enumerate(selected_indices):
        color = cmap(color_index)
        display_label = str(result.source_labels[source_index]).replace("'", "")
        source = np.clip(result.source_abundances[:, source_index].astype(np.float64), y_floor, None)
        predicted = np.clip(result.projected_mass_fractions[:, source_index].astype(np.float64), y_floor, None)

        source_ax.plot(x_values, source, color=color, linewidth=1.3, label=display_label)
        predicted_ax.plot(x_values, predicted, color=color, linewidth=1.3, label=display_label)

    source_ax.set_ylabel("Mass fraction")
    source_ax.set_yscale("log")
    source_ax.set_title(f"Sukhbold source abundances: {species_count} selected species")
    source_ax.grid(True, which="both", alpha=0.25)

    predicted_ax.set_xlabel("Enclosed mass (M_sun)")
    predicted_ax.set_ylabel("Mass fraction")
    predicted_ax.set_xlim(x_values[0], 5.0)
    predicted_ax.set_yscale("log")
    predicted_ax.set_title("Predicted abundances projected onto the selected Sukhbold species")
    predicted_ax.grid(True, which="both", alpha=0.25)

    clipped_count = int(np.count_nonzero(result.clipped))
    if clipped_count:
        figure.text(0.02, 0.02, f"Clipped zones: {clipped_count}", fontsize=9)

    legend = source_ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        borderaxespad=0.0,
        title="Species",
        fontsize=8,
    )
    source_ax.add_artist(legend)
    figure.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot source and predicted Sukhbold abundances in two panels.")
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
    figure = plot_sukhbold_two_panel_comparison(result, y_floor=args.y_floor)

    if args.output is None:
        plt.show()
        return

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200)


if __name__ == "__main__":
    main()

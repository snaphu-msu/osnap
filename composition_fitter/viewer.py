"""Simple matplotlib slider viewer for the Heger02 composition fitter."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path
from typing import Sequence

_mplconfigdir = Path(tempfile.gettempdir()) / "osnap-mplconfig"
_mplconfigdir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mplconfigdir))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider

from .fitter import Heger02CompositionFitter
from .heger02_dataset import DEFAULT_ARTIFACT_PATH, DEFAULT_SOURCE_DIR, build_reduced_artifact

# Normalized from progenitors/sukhbold_2016/s25.0_presn.
DEFAULT_NUCLEI = [
    "nt1",
    "h1",
    "he3",
    "he4",
    "c12",
    "n14",
    "o16",
    "ne20",
    "mg24",
    "si28",
    "s32",
    "ar36",
    "ca40",
    "ti44",
    "cr48",
    "fe52",
    "fe54",
    "ni56",
    "fe56",
    "fe",
]

AXIS_LABELS = ("log10 rho", "log10 T", "Ye")
SLIDER_TITLES = ("rho", "T", "Ye")


class LimitSlider(Slider):
    """Slider with dynamic active bounds drawn over a fixed global axis."""

    def __init__(self, ax, label: str, global_min: float, global_max: float, *, valinit: float):
        self.global_min = float(global_min)
        self.global_max = float(global_max)
        self.active_min = float(global_min)
        self.active_max = float(global_max)
        super().__init__(ax, label, global_min, global_max, valinit=valinit)
        self.ax.set_xlim(self.global_min, self.global_max)
        self._limit_span = None
        self._lower_line = self.ax.axvline(self.active_min, color="tab:red", linewidth=1.0, alpha=0.8)
        self._upper_line = self.ax.axvline(self.active_max, color="tab:red", linewidth=1.0, alpha=0.8)
        self._draw_active_band()

    def _draw_active_band(self) -> None:
        if self._limit_span is not None:
            self._limit_span.remove()
        self._limit_span = self.ax.axvspan(self.active_min, self.active_max, color="tab:green", alpha=0.18)
        self._lower_line.set_xdata([self.active_min, self.active_min])
        self._upper_line.set_xdata([self.active_max, self.active_max])

    def set_active_bounds(self, lower: float, upper: float) -> None:
        self.active_min = max(self.global_min, float(lower))
        self.active_max = min(self.global_max, float(upper))
        if self.active_min > self.active_max:
            midpoint = 0.5 * (self.active_min + self.active_max)
            self.active_min = midpoint
            self.active_max = midpoint
        self._draw_active_band()

    def _value_in_bounds(self, val):
        val = self._stepped_value(val)

        if val <= self.active_min:
            if not self.closedmin:
                return
            val = self.active_min
        elif val >= self.active_max:
            if not self.closedmax:
                return
            val = self.active_max

        if self.slidermin is not None and val <= self.slidermin.val:
            if not self.closedmin:
                return
            val = self.slidermin.val

        if self.slidermax is not None and val >= self.slidermax.val:
            if not self.closedmax:
                return
            val = self.slidermax.val
        return val


def prepare_plot_arrays(prediction) -> tuple[list[str], np.ndarray, np.ndarray]:
    labels = [str(name) for name in prediction.isotopes.tolist()]
    values = prediction.mass_fractions.astype(np.float64)
    lower = np.maximum(values - prediction.q16, 0.0)
    upper = np.maximum(prediction.q84 - values, 0.0)
    return labels, values, np.vstack((lower, upper))


def ensure_artifact(artifact_path: Path, source_dir: Path) -> Path:
    if artifact_path.exists():
        return artifact_path
    return build_reduced_artifact(source_dir, artifact_path)


def stabilize_query_to_limits(
    fitter: Heger02CompositionFitter,
    query: Sequence[float],
    *,
    iterations: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Iteratively clamp a query into the locally valid slider ranges."""

    stabilized = np.asarray(query, dtype=np.float64).copy()
    limits = fitter.local_valid_ranges(stabilized)
    for _ in range(iterations):
        clipped = np.clip(stabilized, limits[:, 0], limits[:, 1])
        if np.allclose(clipped, stabilized, atol=1.0e-10, rtol=0.0):
            return clipped, limits
        stabilized = clipped
        limits = fitter.local_valid_ranges(stabilized)
    stabilized = np.clip(stabilized, limits[:, 0], limits[:, 1])
    return stabilized, fitter.local_valid_ranges(stabilized)


def launch_viewer(
    artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    nuclei: Sequence[str] = DEFAULT_NUCLEI,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
) -> None:
    artifact_path = ensure_artifact(Path(artifact_path), Path(source_dir))
    fitter = Heger02CompositionFitter.load(artifact_path)

    rho_min, temp_min, ye_min = fitter.feature_min
    rho_max, temp_max, ye_max = fitter.feature_max
    initial_query = np.array(
        [(rho_min + rho_max) / 2.0, (temp_min + temp_max) / 2.0, (ye_min + ye_max) / 2.0],
        dtype=np.float64,
    )

    fig, ax = plt.subplots(figsize=(11, 6))
    plt.subplots_adjust(bottom=0.25)

    prediction = fitter.predict(10 ** initial_query[0], 10 ** initial_query[1], initial_query[2], nuclei=nuclei)
    labels, values, yerr = prepare_plot_arrays(prediction)
    bars = ax.bar(labels, values, yerr=yerr, capsize=4)
    ax.set_ylabel("Mass Fraction")
    ax.set_xlabel("Isotope")
    ax.set_ylim(0.0, max(1.0e-12, float(np.max(prediction.q84)) * 1.15))
    title = ax.set_title("Heger02 constrained composition fit")
    status = fig.text(0.02, 0.97, "", ha="left", va="top")

    slider_ax_rho = fig.add_axes([0.15, 0.14, 0.7, 0.03])
    slider_ax_temp = fig.add_axes([0.15, 0.09, 0.7, 0.03])
    slider_ax_ye = fig.add_axes([0.15, 0.04, 0.7, 0.03])

    rho_slider = LimitSlider(slider_ax_rho, "log10 rho", rho_min, rho_max, valinit=initial_query[0])
    temp_slider = LimitSlider(slider_ax_temp, "log10 T", temp_min, temp_max, valinit=initial_query[1])
    ye_slider = LimitSlider(slider_ax_ye, "Ye", ye_min, ye_max, valinit=initial_query[2])
    sliders = [rho_slider, temp_slider, ye_slider]
    _updating = False

    def apply_slider_state(query: np.ndarray, limits: np.ndarray) -> None:
        nonlocal _updating
        _updating = True
        try:
            for axis, slider in enumerate(sliders):
                slider.set_active_bounds(limits[axis, 0], limits[axis, 1])
                if not np.isclose(slider.val, query[axis], atol=1.0e-12, rtol=0.0):
                    slider.set_val(float(query[axis]))
        finally:
            _updating = False

    def redraw(_: float) -> None:
        nonlocal _updating
        if _updating:
            return

        current = np.array([rho_slider.val, temp_slider.val, ye_slider.val], dtype=np.float64)
        stabilized_query, limits = stabilize_query_to_limits(fitter, current)
        apply_slider_state(stabilized_query, limits)

        try:
            result = fitter.predict(
                10 ** stabilized_query[0],
                10 ** stabilized_query[1],
                stabilized_query[2],
                nuclei=nuclei,
            )
        except Exception as exc:
            status.set_text(f"query failed: {exc}")
            fig.canvas.draw_idle()
            return

        effective_query = result.query.astype(np.float64)
        if not np.allclose(effective_query, stabilized_query, atol=1.0e-10, rtol=0.0):
            stabilized_query, limits = stabilize_query_to_limits(fitter, effective_query)
            apply_slider_state(stabilized_query, limits)
            result = fitter.predict(
                10 ** stabilized_query[0],
                10 ** stabilized_query[1],
                stabilized_query[2],
                nuclei=nuclei,
            )

        _, new_values, new_yerr = prepare_plot_arrays(result)

        for bar, value in zip(bars, new_values):
            bar.set_height(float(value))

        for artist in list(ax.lines) + list(ax.collections):
            artist.remove()

        ax.errorbar(
            np.arange(len(new_values)),
            new_values,
            yerr=new_yerr,
            fmt="none",
            ecolor="black",
            elinewidth=1,
            capsize=4,
        )
        ax.set_ylim(0.0, max(1.0e-12, float(np.max(result.q84)) * 1.15))

        title.set_text(
            "Heger02 constrained composition fit "
            f"(rho={10 ** result.query[0]:.3e}, T={10 ** result.query[1]:.3e}, Ye={result.query[2]:.6f})"
        )
        warning_text = " | ".join(result.warnings) if result.warnings else "in-domain query"
        limit_text = " ; ".join(
            f"{SLIDER_TITLES[axis]} in [{limits[axis, 0]:.4f}, {limits[axis, 1]:.4f}]"
            for axis in range(3)
        )
        status.set_text(
            f"neighbors={result.neighbor_count}  Neff={result.effective_sample_size:.1f}  {warning_text}\n{limit_text}"
        )
        fig.canvas.draw_idle()

    redraw(0.0)
    rho_slider.on_changed(redraw)
    temp_slider.on_changed(redraw)
    ye_slider.on_changed(redraw)
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interactive Heger02 composition viewer.")
    parser.add_argument(
        "--artifact",
        type=Path,
        default=DEFAULT_ARTIFACT_PATH,
        help="Path to the cached fitter artifact; built automatically if missing.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Raw progenitor directory used only if the artifact needs to be built.",
    )
    parser.add_argument(
        "--nuclei",
        nargs="+",
        default=DEFAULT_NUCLEI,
        help="Nuclei to plot in the viewer.",
    )
    args = parser.parse_args()
    launch_viewer(args.artifact, nuclei=args.nuclei, source_dir=args.source)


if __name__ == "__main__":
    main()

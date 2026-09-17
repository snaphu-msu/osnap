"""Verification helpers for held-out Heger02 progenitor fits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fitter import Heger02CompositionFitter
from .heger02_dataset import (
    DEFAULT_ARTIFACT_PATH,
    DEFAULT_SOURCE_DIR,
    _build_artifact_arrays,
    _reduce_artifact_arrays,
    load_artifact,
    scan_presn_file,
)


@dataclass(frozen=True)
class HoldoutVerificationResult:
    """Aggregate errors for one held-out Heger02 progenitor file."""

    test_model_name: str
    zone_count: int
    isotope_count: int
    mean_total_variation: float
    median_total_variation: float
    p90_total_variation: float
    max_total_variation: float
    mean_ye_abs_error: float
    max_ye_abs_error: float
    clipped_fraction: float
    warning_count: int
    total_variation_by_zone: np.ndarray
    ye_abs_error_by_zone: np.ndarray


def verify_holdout_model(
    artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
    *,
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    test_model_name: str | None = None,
    max_zones: int | None = None,
    clip: bool = True,
) -> HoldoutVerificationResult:
    """Compare fitter predictions against the actual abundances in a held-out model.

    The actual held-out composition is projected onto the artifact's reduced
    target network before comparison. Per-zone composition error is reported as
    total variation distance: ``0.5 * sum(abs(predicted - actual))``.
    """

    artifact = load_artifact(artifact_path)
    resolved_test_name = _resolve_test_model_name(artifact, test_model_name)
    reduced_net_path = _resolve_reduced_net_path(artifact)
    truth = _load_reduced_presn_file(Path(source_dir) / resolved_test_name, reduced_net_path)

    fitter = Heger02CompositionFitter(artifact)
    if truth["isotopes"].tolist() != fitter.isotopes.tolist():
        raise ValueError("Held-out truth isotope basis does not match the fitter artifact basis.")

    selected = _selected_zone_indices(truth["compositions"].shape[0], max_zones)
    predicted = np.empty((selected.shape[0], fitter.isotopes.shape[0]), dtype=np.float64)
    clipped = np.zeros(selected.shape[0], dtype=bool)
    warning_count = 0

    for output_index, zone_index in enumerate(selected):
        prediction = fitter.predict(
            float(10.0 ** truth["log_rho"][zone_index]),
            float(10.0 ** truth["log_temp"][zone_index]),
            float(truth["ye"][zone_index]),
            clip=clip,
        )
        predicted[output_index] = prediction.mass_fractions.astype(np.float64)
        clipped[output_index] = prediction.clipped
        warning_count += len(prediction.warnings)

    actual = truth["compositions"][selected].astype(np.float64)
    total_variation = 0.5 * np.sum(np.abs(predicted - actual), axis=1)
    ye_abs_error = np.abs((predicted @ fitter.z_over_a.astype(np.float64)) - truth["ye"][selected].astype(np.float64))

    return HoldoutVerificationResult(
        test_model_name=resolved_test_name,
        zone_count=int(selected.shape[0]),
        isotope_count=int(fitter.isotopes.shape[0]),
        mean_total_variation=float(np.mean(total_variation)),
        median_total_variation=float(np.median(total_variation)),
        p90_total_variation=float(np.quantile(total_variation, 0.90)),
        max_total_variation=float(np.max(total_variation)),
        mean_ye_abs_error=float(np.mean(ye_abs_error)),
        max_ye_abs_error=float(np.max(ye_abs_error)),
        clipped_fraction=float(np.mean(clipped)),
        warning_count=int(warning_count),
        total_variation_by_zone=total_variation.astype(np.float32),
        ye_abs_error_by_zone=ye_abs_error.astype(np.float32),
    )


def _load_reduced_presn_file(path: Path, reduced_net_path: Path) -> dict[str, np.ndarray]:
    schema = scan_presn_file(path)
    full_arrays = _build_artifact_arrays([schema])
    return _reduce_artifact_arrays(full_arrays, reduced_net_path)


def _resolve_test_model_name(artifact: dict[str, np.ndarray], test_model_name: str | None) -> str:
    if test_model_name is not None:
        return test_model_name

    names = artifact.get("split_test_progenitor_names")
    if names is None or names.size != 1:
        raise ValueError("Artifact must contain exactly one split test progenitor name.")
    return str(names[0])


def _resolve_reduced_net_path(artifact: dict[str, np.ndarray]) -> Path:
    paths = artifact.get("reduced_net_path")
    if paths is None or paths.size != 1:
        raise ValueError("Holdout verification requires a reduced-network artifact.")
    return Path(str(paths[0]))


def _selected_zone_indices(zone_count: int, max_zones: int | None) -> np.ndarray:
    if max_zones is None or max_zones >= zone_count:
        return np.arange(zone_count, dtype=np.int64)
    if max_zones <= 0:
        raise ValueError("max_zones must be positive when provided.")
    return np.linspace(0, zone_count - 1, max_zones, dtype=np.int64)

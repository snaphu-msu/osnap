"""Repair reduced-network progenitor compositions against file-column Ye."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize

from .isotopes import parse_isotope
from .sukhbold_profile import SukhboldProfile

DEFAULT_UNSTABLE_ISOTOPES = ("ti44", "cr48", "fe52", "ni56")
DEFAULT_NEGATIVE_TOLERANCE = 1.0e-14
DEFAULT_YE_TOLERANCE = 1.0e-10
_LABEL_ALIASES = {
    "n": "nt1",
    "neut": "nt1",
    "neutrons": "nt1",
    "p": "h1",
    "prot": "h1",
}


@dataclass(frozen=True)
class RepairedComposition:
    """Canonical reduced-network abundances before and after Ye repair."""

    labels: np.ndarray
    z_over_a: np.ndarray
    original_mass_fractions: np.ndarray
    edited_mass_fractions: np.ndarray
    reference_mass_fractions: np.ndarray
    mass_fractions: np.ndarray
    target_ye: np.ndarray
    diagnostics: dict[str, np.ndarray]


@dataclass(frozen=True)
class RepairedSukhboldComposition:
    """Ye-repaired composition for a parsed Sukhbold progenitor profile."""

    profile: SukhboldProfile
    labels: np.ndarray
    z_over_a: np.ndarray
    original_mass_fractions: np.ndarray
    edited_mass_fractions: np.ndarray
    reference_mass_fractions: np.ndarray
    mass_fractions: np.ndarray
    target_ye: np.ndarray
    diagnostics: dict[str, np.ndarray]

    @property
    def path(self) -> Path:
        return self.profile.path

    @property
    def enclosed_mass_g(self) -> np.ndarray:
        return self.profile.enclosed_mass_g


def repair_reduced_abundances(
    labels: Sequence[str],
    abundances: np.ndarray,
    target_ye: np.ndarray | Sequence[float],
    *,
    unstable_isotopes: Sequence[str] = DEFAULT_UNSTABLE_ISOTOPES,
    ye_tolerance: float = DEFAULT_YE_TOLERANCE,
    negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE,
) -> RepairedComposition:
    """Canonicalize and repair reduced-network abundances to match target ``Ye``.

    The ambiguous Sukhbold ``fe`` bucket is treated as ``fe56`` and merged into
    any explicit ``fe56`` column before selected unstable isotopes are zeroed.
    """

    source = np.asarray(abundances, dtype=np.float64)
    if source.ndim == 1:
        source = source.reshape(1, -1)
    if source.ndim != 2:
        raise ValueError("abundances must be a one- or two-dimensional array.")
    if source.shape[1] != len(labels):
        raise ValueError("abundances column count must match labels.")
    if not np.all(np.isfinite(source)):
        raise ValueError("abundances contain non-finite values.")
    if np.any(source < -negative_tolerance):
        raise ValueError("abundances contain negative values below tolerance.")
    source = np.where(source < 0.0, 0.0, source)

    target = np.asarray(target_ye, dtype=np.float64)
    if target.ndim == 0:
        target = np.full(source.shape[0], float(target), dtype=np.float64)
    if target.ndim != 1 or target.shape[0] != source.shape[0]:
        raise ValueError("target_ye must be scalar or have one value per abundance row.")
    if not np.all(np.isfinite(target)):
        raise ValueError("target_ye contains non-finite values.")

    canonical_labels, original, merged_fe_mass = _canonicalize_abundances(labels, source)
    if not canonical_labels:
        raise ValueError("No reduced-network abundance labels were provided.")

    z_over_a = np.array([parse_isotope(label).z_over_a for label in canonical_labels], dtype=np.float64)
    unstable = {_canonical_label(label) for label in unstable_isotopes}
    unstable_indices = np.array([index for index, label in enumerate(canonical_labels) if label in unstable], dtype=np.int64)

    edited = original.copy()
    if unstable_indices.size:
        edited[:, unstable_indices] = 0.0

    repaired = np.empty_like(edited)
    reference = np.empty_like(edited)
    diagnostics = _empty_diagnostics(source.shape[0])
    diagnostics["original_sum"] = original.sum(axis=1)
    diagnostics["edited_sum_before_projection"] = edited.sum(axis=1)
    diagnostics["original_composition_ye"] = _composition_ye(original, z_over_a)
    diagnostics["edited_composition_ye"] = _composition_ye(edited, z_over_a)
    diagnostics["zeroed_unstable_mass"] = (
        original[:, unstable_indices].sum(axis=1) if unstable_indices.size else np.zeros(source.shape[0], dtype=np.float64)
    )
    diagnostics["merged_fe_mass"] = merged_fe_mass

    bounds = [
        (0.0, 0.0) if label in unstable else (0.0, 1.0)
        for label in canonical_labels
    ]

    for row_index in range(source.shape[0]):
        repaired[row_index], reference[row_index], success = _repair_row(
            edited[row_index],
            z_over_a,
            float(target[row_index]),
            bounds,
            ye_tolerance=ye_tolerance,
        )
        diagnostics["optimizer_success"][row_index] = success

    repaired_ye = repaired @ z_over_a
    diagnostics["repaired_composition_ye"] = repaired_ye
    diagnostics["sum_error"] = repaired.sum(axis=1) - 1.0
    diagnostics["ye_error"] = repaired_ye - target
    diagnostics["projection_l2_delta"] = np.linalg.norm(repaired - reference, axis=1)

    return RepairedComposition(
        labels=np.asarray(canonical_labels),
        z_over_a=z_over_a,
        original_mass_fractions=original,
        edited_mass_fractions=edited,
        reference_mass_fractions=reference,
        mass_fractions=repaired,
        target_ye=target,
        diagnostics=diagnostics,
    )


def repair_sukhbold_profile(
    profile: SukhboldProfile,
    *,
    unstable_isotopes: Sequence[str] = DEFAULT_UNSTABLE_ISOTOPES,
    ye_tolerance: float = DEFAULT_YE_TOLERANCE,
    negative_tolerance: float = DEFAULT_NEGATIVE_TOLERANCE,
) -> RepairedSukhboldComposition:
    """Repair a parsed Sukhbold profile's reduced-network composition."""

    repaired = repair_reduced_abundances(
        profile.selector_labels.tolist(),
        profile.source_abundances,
        profile.ye,
        unstable_isotopes=unstable_isotopes,
        ye_tolerance=ye_tolerance,
        negative_tolerance=negative_tolerance,
    )
    return RepairedSukhboldComposition(
        profile=profile,
        labels=repaired.labels,
        z_over_a=repaired.z_over_a,
        original_mass_fractions=repaired.original_mass_fractions,
        edited_mass_fractions=repaired.edited_mass_fractions,
        reference_mass_fractions=repaired.reference_mass_fractions,
        mass_fractions=repaired.mass_fractions,
        target_ye=repaired.target_ye,
        diagnostics=repaired.diagnostics,
    )


def _canonicalize_abundances(labels: Sequence[str], source: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
    canonical_labels: list[str] = []
    canonical_columns: list[np.ndarray] = []
    label_to_index: dict[str, int] = {}
    merged_fe_mass = np.zeros(source.shape[0], dtype=np.float64)

    for column_index, raw_label in enumerate(labels):
        normalized = _normalize_label(raw_label)
        canonical = "fe56" if normalized == "fe" else _LABEL_ALIASES.get(normalized, normalized)
        parse_isotope(canonical)

        if normalized == "fe":
            merged_fe_mass += source[:, column_index]

        existing = label_to_index.get(canonical)
        if existing is None:
            label_to_index[canonical] = len(canonical_labels)
            canonical_labels.append(canonical)
            canonical_columns.append(source[:, column_index].copy())
        else:
            canonical_columns[existing] += source[:, column_index]

    canonical = np.column_stack(canonical_columns) if canonical_columns else np.empty((source.shape[0], 0))
    return canonical_labels, canonical, merged_fe_mass


def _canonical_label(label: str) -> str:
    normalized = _normalize_label(label)
    if normalized == "fe":
        return "fe56"
    return _LABEL_ALIASES.get(normalized, normalized)


def _normalize_label(label: str) -> str:
    return str(label).strip().strip("'").strip('"').lower()


def _empty_diagnostics(row_count: int) -> dict[str, np.ndarray]:
    return {
        "original_sum": np.empty(row_count, dtype=np.float64),
        "edited_sum_before_projection": np.empty(row_count, dtype=np.float64),
        "original_composition_ye": np.empty(row_count, dtype=np.float64),
        "edited_composition_ye": np.empty(row_count, dtype=np.float64),
        "repaired_composition_ye": np.empty(row_count, dtype=np.float64),
        "sum_error": np.empty(row_count, dtype=np.float64),
        "ye_error": np.empty(row_count, dtype=np.float64),
        "zeroed_unstable_mass": np.empty(row_count, dtype=np.float64),
        "merged_fe_mass": np.empty(row_count, dtype=np.float64),
        "projection_l2_delta": np.empty(row_count, dtype=np.float64),
        "optimizer_success": np.zeros(row_count, dtype=np.int8),
    }


def _composition_ye(values: np.ndarray, z_over_a: np.ndarray) -> np.ndarray:
    mass_sum = values.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ye = (values @ z_over_a) / mass_sum
    ye[mass_sum <= 0.0] = np.nan
    return ye


def _repair_row(
    edited: np.ndarray,
    z_over_a: np.ndarray,
    target_ye: float,
    bounds: Sequence[tuple[float, float]],
    *,
    ye_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, int]:
    allowed = np.array([index for index, bound in enumerate(bounds) if bound[1] > 0.0], dtype=np.int64)
    if allowed.size == 0:
        raise ValueError("No non-fixed species are available for Ye repair.")

    allowed_ye = z_over_a[allowed]
    if target_ye < allowed_ye.min() - ye_tolerance or target_ye > allowed_ye.max() + ye_tolerance:
        raise ValueError(
            f"Target Ye {target_ye:.12g} is outside the available reduced-network range "
            f"[{allowed_ye.min():.12g}, {allowed_ye.max():.12g}]."
        )

    initial = _feasible_initial_guess(z_over_a, target_ye, allowed, ye_tolerance=ye_tolerance)
    edited_sum = float(edited.sum())
    reference = edited / edited_sum if edited_sum > 0.0 else initial.copy()

    if _constraint_residual(initial, z_over_a, target_ye) <= ye_tolerance and allowed.size == 1:
        return initial, reference, 1

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones_like(x)},
        {"type": "eq", "fun": lambda x: np.dot(x, z_over_a) - target_ye, "jac": lambda x: z_over_a},
    )
    result = minimize(
        fun=lambda x: 0.5 * np.sum((x - reference) ** 2),
        x0=initial,
        jac=lambda x: x - reference,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": min(1.0e-12, ye_tolerance), "maxiter": 200},
    )
    if not result.success:
        raise RuntimeError(f"Failed to repair reduced composition to Ye constraint: {result.message}")

    values = np.asarray(result.x, dtype=np.float64)
    residual = _constraint_residual(values, z_over_a, target_ye)
    if residual > max(ye_tolerance, 1.0e-10):
        raise RuntimeError(f"Repaired composition residual {residual:.3e} exceeds tolerance.")
    return values, reference, 1


def _feasible_initial_guess(
    z_over_a: np.ndarray,
    target_ye: float,
    allowed: np.ndarray,
    *,
    ye_tolerance: float,
) -> np.ndarray:
    guess = np.zeros(z_over_a.shape[0], dtype=np.float64)
    allowed_ye = z_over_a[allowed]
    close = np.flatnonzero(np.isclose(allowed_ye, target_ye, atol=ye_tolerance))
    if close.size:
        guess[int(allowed[int(close[0])])] = 1.0
        return guess

    below_slots = np.flatnonzero(allowed_ye < target_ye)
    above_slots = np.flatnonzero(allowed_ye > target_ye)
    if below_slots.size == 0 or above_slots.size == 0:
        raise ValueError(f"Could not bracket target Ye {target_ye:.12g} with available species.")

    below_slot = int(below_slots[np.argmax(allowed_ye[below_slots])])
    above_slot = int(above_slots[np.argmin(allowed_ye[above_slots])])
    below_index = int(allowed[below_slot])
    above_index = int(allowed[above_slot])
    low_ye = float(z_over_a[below_index])
    high_ye = float(z_over_a[above_index])
    high_weight = (target_ye - low_ye) / (high_ye - low_ye)
    high_weight = float(np.clip(high_weight, 0.0, 1.0))
    guess[below_index] = 1.0 - high_weight
    guess[above_index] = high_weight
    return guess


def _constraint_residual(values: np.ndarray, z_over_a: np.ndarray, target_ye: float) -> float:
    return max(abs(float(np.sum(values) - 1.0)), abs(float(np.dot(values, z_over_a) - target_ye)))

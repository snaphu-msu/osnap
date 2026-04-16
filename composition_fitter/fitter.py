"""Query interface for constrained local fits over the cached Heger02 artifact."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, TYPE_CHECKING, Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree

from .heger02_dataset import DEFAULT_ARTIFACT_PATH, load_artifact

if TYPE_CHECKING:
    from .sukhbold_profile import SukhboldProfile


@dataclass
class Prediction:
    """Returned data for a single fit query."""

    isotopes: np.ndarray
    mass_fractions: np.ndarray
    q16: np.ndarray
    q50: np.ndarray
    q84: np.ndarray
    query: np.ndarray
    effective_sample_size: float
    neighbor_count: int
    clipped: bool
    warnings: list[str]

    def as_dict(self) -> dict[str, object]:
        return {
            "isotopes": self.isotopes,
            "mass_fractions": self.mass_fractions,
            "q16": self.q16,
            "q50": self.q50,
            "q84": self.q84,
            "query": self.query,
            "effective_sample_size": self.effective_sample_size,
            "neighbor_count": self.neighbor_count,
            "clipped": self.clipped,
            "warnings": self.warnings,
        }


@dataclass
class _PredictionContext:
    """Internal state for one query before optional isotope grouping."""

    query: np.ndarray
    clipped: bool
    warnings: list[str]
    weights: np.ndarray
    local_compositions: np.ndarray
    mass_fractions: np.ndarray
    effective_sample_size: float
    neighbor_count: int


@dataclass
class ProfilePrediction:
    """Returned data for Sukhbold radius-profile predictions."""

    enclosed_mass_g: np.ndarray
    radius_cm: np.ndarray
    log_radius_cm: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    ye: np.ndarray
    predicted_ye: np.ndarray
    source_labels: np.ndarray
    selector_labels: np.ndarray
    source_abundances: np.ndarray
    predicted_isotopes: np.ndarray
    predicted_mass_fractions: np.ndarray
    predicted_q16: np.ndarray
    predicted_q50: np.ndarray
    predicted_q84: np.ndarray
    projected_mass_fractions: np.ndarray
    projected_q16: np.ndarray
    projected_q50: np.ndarray
    projected_q84: np.ndarray
    clipped: np.ndarray
    warnings: tuple[tuple[str, ...], ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "enclosed_mass_g": self.enclosed_mass_g,
            "radius_cm": self.radius_cm,
            "log_radius_cm": self.log_radius_cm,
            "density": self.density,
            "temperature": self.temperature,
            "ye": self.ye,
            "predicted_ye": self.predicted_ye,
            "source_labels": self.source_labels,
            "selector_labels": self.selector_labels,
            "source_abundances": self.source_abundances,
            "predicted_isotopes": self.predicted_isotopes,
            "predicted_mass_fractions": self.predicted_mass_fractions,
            "predicted_q16": self.predicted_q16,
            "predicted_q50": self.predicted_q50,
            "predicted_q84": self.predicted_q84,
            "projected_mass_fractions": self.projected_mass_fractions,
            "projected_q16": self.projected_q16,
            "projected_q50": self.projected_q50,
            "projected_q84": self.projected_q84,
            "clipped": self.clipped,
            "warnings": self.warnings,
        }


class Heger02CompositionFitter:
    """Local kernel fitter with exact Ye constraints."""

    def __init__(
        self,
        artifact: dict[str, np.ndarray],
        *,
        initial_k: int = 64,
        max_k: int | None = None,
        limit_k: int = 512,
        ye_tolerance: float = 1.0e-6,
    ) -> None:
        self.compositions = artifact["compositions"].astype(np.float64, copy=False)
        self.feature_matrix = artifact["feature_matrix"].astype(np.float64, copy=False)
        self.feature_mean = artifact["feature_mean"].astype(np.float64, copy=False)
        self.feature_std = artifact["feature_std"].astype(np.float64, copy=False)
        self.feature_min = artifact["feature_min"].astype(np.float64, copy=False)
        self.feature_max = artifact["feature_max"].astype(np.float64, copy=False)
        self.ye = artifact["ye"].astype(np.float64, copy=False)
        self.z_over_a = artifact["z_over_a"].astype(np.float64, copy=False)
        self.isotopes = artifact["isotopes"]
        self.initial_k = max(2, min(initial_k, self.feature_matrix.shape[0]))
        self.max_k = max_k
        self.limit_k = max(2, min(limit_k, self.feature_matrix.shape[0]))
        self.ye_tolerance = ye_tolerance
        self.isotope_to_index = {str(name): index for index, name in enumerate(self.isotopes.tolist())}

        standardized = (self.feature_matrix - self.feature_mean) / self.feature_std
        self._standardized_features = standardized
        self._tree = cKDTree(standardized)
        self._limit_pair_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}
        self._limit_trees = {
            axis: cKDTree(standardized[:, pair_axes])
            for axis, pair_axes in self._limit_pair_axes.items()
        }

    @classmethod
    def load(
        cls,
        artifact_path: Path | str = DEFAULT_ARTIFACT_PATH,
        *,
        initial_k: int = 64,
        max_k: int | None = None,
        limit_k: int = 512,
        ye_tolerance: float = 1.0e-6,
    ) -> "Heger02CompositionFitter":
        return cls(
            load_artifact(artifact_path),
            initial_k=initial_k,
            max_k=max_k,
            limit_k=limit_k,
            ye_tolerance=ye_tolerance,
        )

    def axis_limits(self, query: Sequence[float], axis: int) -> tuple[float, float]:
        """Return the local valid range for one axis given the other two query coordinates."""

        query_array = np.asarray(query, dtype=np.float64)
        pair_axes = self._limit_pair_axes[axis]
        standardized_pair = (query_array[list(pair_axes)] - self.feature_mean[list(pair_axes)]) / self.feature_std[
            list(pair_axes)
        ]
        distances, indices = self._limit_trees[axis].query(standardized_pair, k=self.limit_k)
        indices = np.atleast_1d(indices).astype(np.int64)
        axis_values = self.feature_matrix[indices, axis]
        return float(axis_values.min()), float(axis_values.max())

    def local_valid_ranges(self, query: Sequence[float]) -> np.ndarray:
        """Return per-axis local valid ranges as ``[[min, max], ...]``."""

        query_array = np.asarray(query, dtype=np.float64)
        return np.array([self.axis_limits(query_array, axis) for axis in range(3)], dtype=np.float64)

    def predict(
        self,
        density: float,
        temperature: float,
        ye: float,
        nuclei: Sequence[str] | None = None,
        *,
        clip: bool = True,
    ) -> Prediction:
        context = self._predict_context(density, temperature, ye, clip=clip)
        isotopes, mass_fractions, q16, q50, q84 = self._summarize_selection(context, nuclei)

        return Prediction(
            isotopes=isotopes,
            mass_fractions=mass_fractions.astype(np.float32),
            q16=q16.astype(np.float32),
            q50=q50.astype(np.float32),
            q84=q84.astype(np.float32),
            query=context.query.astype(np.float32),
            effective_sample_size=context.effective_sample_size,
            neighbor_count=context.neighbor_count,
            clipped=context.clipped,
            warnings=context.warnings,
        )

    def predict_profile(
        self,
        profile: SukhboldProfile | Path | str,
        *,
        clip: bool = True,
    ) -> ProfilePrediction:
        if isinstance(profile, (str, Path)):
            from .sukhbold_profile import read_sukhbold_profile

            profile = read_sukhbold_profile(profile)

        groups = self._resolve_nuclei_groups(profile.selector_labels.tolist())
        zone_count = profile.radius_cm.shape[0]
        isotope_count = self.isotopes.shape[0]
        projection_count = len(groups)

        predicted_mass_fractions = np.empty((zone_count, isotope_count), dtype=np.float32)
        predicted_q16 = np.empty_like(predicted_mass_fractions)
        predicted_q50 = np.empty_like(predicted_mass_fractions)
        predicted_q84 = np.empty_like(predicted_mass_fractions)
        projected_mass_fractions = np.empty((zone_count, projection_count), dtype=np.float32)
        projected_q16 = np.empty_like(projected_mass_fractions)
        projected_q50 = np.empty_like(projected_mass_fractions)
        projected_q84 = np.empty_like(projected_mass_fractions)
        predicted_ye = np.empty(zone_count, dtype=np.float32)
        clipped_mask = np.zeros(zone_count, dtype=bool)
        warnings_by_zone: list[tuple[str, ...]] = []

        for zone_index in range(zone_count):
            context = self._predict_context(
                float(profile.density[zone_index]),
                float(profile.temperature[zone_index]),
                float(profile.ye[zone_index]),
                clip=clip,
            )
            full_isotopes, full_mass_fractions, full_q16, full_q50, full_q84 = self._summarize_selection(context, None)
            (
                _projected_labels,
                zone_projected_mass_fractions,
                zone_projected_q16,
                zone_projected_q50,
                zone_projected_q84,
            ) = self._summarize_groups(context, groups)

            predicted_mass_fractions[zone_index] = full_mass_fractions.astype(np.float32)
            predicted_q16[zone_index] = full_q16.astype(np.float32)
            predicted_q50[zone_index] = full_q50.astype(np.float32)
            predicted_q84[zone_index] = full_q84.astype(np.float32)
            predicted_ye[zone_index] = np.float32(np.dot(full_mass_fractions.astype(np.float64), self.z_over_a))
            projected_mass_fractions[zone_index] = zone_projected_mass_fractions.astype(np.float32)
            projected_q16[zone_index] = zone_projected_q16.astype(np.float32)
            projected_q50[zone_index] = zone_projected_q50.astype(np.float32)
            projected_q84[zone_index] = zone_projected_q84.astype(np.float32)
            clipped_mask[zone_index] = context.clipped
            warnings_by_zone.append(tuple(context.warnings))

            if zone_index == 0 and not np.array_equal(full_isotopes, self.isotopes):
                raise RuntimeError("Profile prediction isotope basis drifted from the fitter artifact.")

        return ProfilePrediction(
            enclosed_mass_g=profile.enclosed_mass_g.astype(np.float32, copy=False),
            radius_cm=profile.radius_cm.astype(np.float32, copy=False),
            log_radius_cm=profile.log_radius_cm.astype(np.float32, copy=False),
            density=profile.density.astype(np.float32, copy=False),
            temperature=profile.temperature.astype(np.float32, copy=False),
            ye=profile.ye.astype(np.float32, copy=False),
            predicted_ye=predicted_ye,
            source_labels=profile.source_labels.copy(),
            selector_labels=profile.selector_labels.copy(),
            source_abundances=profile.source_abundances.astype(np.float32, copy=False),
            predicted_isotopes=self.isotopes.copy(),
            predicted_mass_fractions=predicted_mass_fractions,
            predicted_q16=predicted_q16,
            predicted_q50=predicted_q50,
            predicted_q84=predicted_q84,
            projected_mass_fractions=projected_mass_fractions,
            projected_q16=projected_q16,
            projected_q50=projected_q50,
            projected_q84=projected_q84,
            clipped=clipped_mask,
            warnings=tuple(warnings_by_zone),
        )

    def _predict_context(
        self,
        density: float,
        temperature: float,
        ye: float,
        *,
        clip: bool,
    ) -> _PredictionContext:
        query, clipped, warnings = self._normalize_query(density, temperature, ye, clip=clip)
        neighbor_indices, distances, target_ye, warnings = self._select_neighbors(query, warnings, clip=clip)
        effective_query = query.copy()
        effective_query[2] = target_ye
        prior_weights = self._kernel_prior(distances)
        neighbor_ye = self.ye[neighbor_indices]
        weights = self._solve_constrained_weights(prior_weights, neighbor_ye, target_ye, distances)

        local_compositions = self.compositions[neighbor_indices]
        mass_fractions = weights @ local_compositions
        mass_fractions = np.clip(mass_fractions, 0.0, None)
        mass_sum = mass_fractions.sum()
        if mass_sum <= 0.0:
            raise RuntimeError("Constrained fit produced a non-positive composition sum.")
        mass_fractions /= mass_sum

        ye_from_mass_fractions = float(mass_fractions @ self.z_over_a)
        if abs(ye_from_mass_fractions - target_ye) > self.ye_tolerance:
            warnings.append(
                f"Output Ye drifted by {abs(ye_from_mass_fractions - target_ye):.3e}; applying correction."
            )
            mass_fractions = self._project_to_constraints(mass_fractions, target_ye)

        effective_sample_size = float(1.0 / np.sum(weights**2))
        return _PredictionContext(
            query=effective_query.astype(np.float64, copy=False),
            clipped=clipped,
            warnings=warnings,
            weights=weights.astype(np.float64, copy=False),
            local_compositions=local_compositions.astype(np.float64, copy=False),
            mass_fractions=mass_fractions.astype(np.float64, copy=False),
            effective_sample_size=effective_sample_size,
            neighbor_count=len(neighbor_indices),
        )

    def _summarize_selection(
        self,
        context: _PredictionContext,
        nuclei: Sequence[str] | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if nuclei is None:
            q16, q50, q84 = self._weighted_quantiles(context.local_compositions, context.weights, (0.16, 0.50, 0.84))
            return self.isotopes, context.mass_fractions, q16, q50, q84

        groups = self._resolve_nuclei_groups(nuclei)
        return self._summarize_groups(context, groups)

    def _summarize_groups(
        self,
        context: _PredictionContext,
        groups: Sequence[tuple[str, np.ndarray]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        isotopes = np.array([label for label, _ in groups])
        grouped_local_compositions = np.column_stack(
            [context.local_compositions[:, indices].sum(axis=1) for _, indices in groups]
        )
        mass_fractions = np.array(
            [context.mass_fractions[indices].sum() for _, indices in groups],
            dtype=np.float64,
        )
        q16, q50, q84 = self._weighted_quantiles(grouped_local_compositions, context.weights, (0.16, 0.50, 0.84))
        return isotopes, mass_fractions, q16, q50, q84

    def _normalize_query(
        self,
        density: float,
        temperature: float,
        ye: float,
        *,
        clip: bool,
    ) -> tuple[np.ndarray, bool, list[str]]:
        if density <= 0.0:
            raise ValueError("density must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")

        query = np.array([np.log10(density), np.log10(temperature), ye], dtype=np.float64)
        warnings: list[str] = []
        clipped = False
        if np.any(query < self.feature_min) or np.any(query > self.feature_max):
            if not clip:
                raise ValueError(
                    "Query lies outside the fitted domain: "
                    f"query={query}, min={self.feature_min}, max={self.feature_max}"
                )
            clipped = True
            clipped_query = np.clip(query, self.feature_min, self.feature_max)
            warnings.append(f"Query clipped from {query.tolist()} to {clipped_query.tolist()}.")
            query = clipped_query
        return query, clipped, warnings

    def _select_neighbors(
        self,
        query: np.ndarray,
        warnings: list[str],
        *,
        clip: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, list[str]]:
        standardized_query = (query - self.feature_mean) / self.feature_std
        target_ye = float(query[2])
        total_points = self._standardized_features.shape[0]
        max_k = total_points if self.max_k is None else min(self.max_k, total_points)

        k = self.initial_k
        while True:
            distances, indices = self._tree.query(standardized_query, k=k)
            distances = np.atleast_1d(distances).astype(np.float64)
            indices = np.atleast_1d(indices).astype(np.int64)
            neighbor_ye = self.ye[indices]
            feasible = self._is_ye_feasible(neighbor_ye, target_ye)
            if feasible or k >= max_k:
                break
            k = min(max_k, k * 2)

        if not self._is_ye_feasible(self.ye[indices], target_ye):
            if not clip:
                raise ValueError("Unable to bracket the requested Ye with available neighbors.")
            clipped_target = float(np.clip(target_ye, self.ye[indices].min(), self.ye[indices].max()))
            warnings.append(f"Local Ye clipped from {target_ye:.8f} to {clipped_target:.8f}.")
            target_ye = clipped_target

        return indices, distances, target_ye, warnings

    def _is_ye_feasible(self, neighbor_ye: np.ndarray, target_ye: float) -> bool:
        if np.any(np.isclose(neighbor_ye, target_ye, atol=self.ye_tolerance)):
            return True
        return bool(neighbor_ye.min() - self.ye_tolerance <= target_ye <= neighbor_ye.max() + self.ye_tolerance)

    def _kernel_prior(self, distances: np.ndarray) -> np.ndarray:
        if distances.shape[0] == 1:
            return np.array([1.0], dtype=np.float64)

        bandwidth = max(float(distances[-1]), 1.0e-12)
        weights = np.exp(-0.5 * (distances / bandwidth) ** 2)
        weights_sum = weights.sum()
        if weights_sum <= 0.0:
            return np.full_like(weights, 1.0 / weights.shape[0], dtype=np.float64)
        return weights / weights_sum

    def _solve_constrained_weights(
        self,
        prior_weights: np.ndarray,
        neighbor_ye: np.ndarray,
        target_ye: float,
        distances: np.ndarray,
    ) -> np.ndarray:
        initial = self._initial_feasible_weights(prior_weights, neighbor_ye, target_ye, distances)
        if self._weights_are_feasible(initial, neighbor_ye, target_ye):
            weights = initial
        else:
            weights = self._project_prior_to_feasible_simplex(prior_weights, neighbor_ye, target_ye)
            if weights is None:
                weights = initial

        weights = np.clip(weights, 0.0, None)
        if weights.sum() <= 0.0:
            raise RuntimeError("Constrained neighbor solve produced a zero-sum weight vector.")
        weights /= weights.sum()
        return weights

    def _project_prior_to_feasible_simplex(
        self,
        prior_weights: np.ndarray,
        neighbor_ye: np.ndarray,
        target_ye: float,
    ) -> np.ndarray | None:
        active = np.ones(prior_weights.shape[0], dtype=bool)
        best_candidate: np.ndarray | None = None
        best_objective = np.inf

        for _ in range(prior_weights.shape[0]):
            active_indices = np.flatnonzero(active)
            if active_indices.size < 2:
                break

            active_ye = neighbor_ye[active_indices]
            if not self._is_ye_feasible(active_ye, target_ye):
                break

            projected = self._project_to_affine_constraints(
                prior_weights[active_indices],
                active_ye,
                target_ye,
            )
            if projected is None:
                break

            candidate = np.zeros_like(prior_weights)
            candidate[active_indices] = projected
            objective = 0.5 * np.sum((candidate - prior_weights) ** 2)

            if np.all(projected >= -1.0e-12):
                candidate = np.clip(candidate, 0.0, None)
                candidate /= candidate.sum()
                if self._weights_are_feasible(candidate, neighbor_ye, target_ye):
                    return candidate

            if objective < best_objective:
                best_candidate = candidate
                best_objective = objective

            worst_local = int(np.argmin(projected))
            active[active_indices[worst_local]] = False

        if best_candidate is None:
            return None
        best_candidate = np.clip(best_candidate, 0.0, None)
        if best_candidate.sum() <= 0.0:
            return None
        best_candidate /= best_candidate.sum()
        if self._weights_are_feasible(best_candidate, neighbor_ye, target_ye):
            return best_candidate
        return None

    def _project_to_affine_constraints(
        self,
        prior_weights: np.ndarray,
        neighbor_ye: np.ndarray,
        target_ye: float,
    ) -> np.ndarray | None:
        ones = np.ones_like(prior_weights)
        constraint_matrix = np.vstack((ones, neighbor_ye))
        gram = constraint_matrix @ constraint_matrix.T
        rhs = constraint_matrix @ prior_weights - np.array([1.0, target_ye], dtype=np.float64)

        try:
            lagrange = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            return None
        return prior_weights - constraint_matrix.T @ lagrange

    def _initial_feasible_weights(
        self,
        prior_weights: np.ndarray,
        neighbor_ye: np.ndarray,
        target_ye: float,
        distances: np.ndarray,
    ) -> np.ndarray:
        close = np.where(np.isclose(neighbor_ye, target_ye, atol=self.ye_tolerance))[0]
        if close.size:
            weights = np.zeros_like(prior_weights)
            local = prior_weights[close]
            local_sum = local.sum()
            weights[close] = local / local_sum if local_sum > 0.0 else 1.0 / close.size
            return weights

        below = np.where(neighbor_ye < target_ye)[0]
        above = np.where(neighbor_ye > target_ye)[0]
        if below.size == 0 or above.size == 0:
            best = int(np.argmin(np.abs(neighbor_ye - target_ye)))
            weights = np.zeros_like(prior_weights)
            weights[best] = 1.0
            return weights

        below_index = below[np.argmin(distances[below])]
        above_index = above[np.argmin(distances[above])]
        ye_below = neighbor_ye[below_index]
        ye_above = neighbor_ye[above_index]
        mix_above = (target_ye - ye_below) / (ye_above - ye_below)
        mix_above = float(np.clip(mix_above, 0.0, 1.0))

        weights = np.zeros_like(prior_weights)
        weights[below_index] = 1.0 - mix_above
        weights[above_index] = mix_above
        return weights

    def _weights_are_feasible(self, weights: np.ndarray, neighbor_ye: np.ndarray, target_ye: float) -> bool:
        return bool(
            np.all(weights >= -1.0e-10)
            and abs(np.sum(weights) - 1.0) <= 1.0e-8
            and abs(np.dot(weights, neighbor_ye) - target_ye) <= self.ye_tolerance
        )

    def _project_to_constraints(self, values: np.ndarray, target_ye: float) -> np.ndarray:
        guess = np.clip(values, 0.0, None)
        if guess.sum() <= 0.0:
            guess = np.full_like(values, 1.0 / values.shape[0])
        else:
            guess /= guess.sum()

        constraints = (
            {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones_like(x)},
            {
                "type": "eq",
                "fun": lambda x: np.dot(x, self.z_over_a) - target_ye,
                "jac": lambda x: self.z_over_a,
            },
        )
        bounds = [(0.0, 1.0)] * values.shape[0]
        result = minimize(
            fun=lambda x: 0.5 * np.sum((x - values) ** 2),
            x0=guess,
            jac=lambda x: x - values,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1.0e-12, "maxiter": 400, "disp": False},
        )
        if not result.success:
            raise RuntimeError(f"Failed to project composition to Ye constraint: {result.message}")
        corrected = np.clip(result.x, 0.0, None)
        corrected /= corrected.sum()
        return corrected

    def _weighted_quantiles(
        self,
        values: np.ndarray,
        weights: np.ndarray,
        probabilities: Iterable[float],
    ) -> tuple[np.ndarray, ...]:
        order = np.argsort(values, axis=0)
        sorted_values = np.take_along_axis(values, order, axis=0)
        repeated_weights = np.broadcast_to(weights[:, None], values.shape)
        sorted_weights = np.take_along_axis(repeated_weights, order, axis=0)
        cumulative = np.cumsum(sorted_weights, axis=0)
        cumulative /= cumulative[-1, :]

        quantiles = []
        for probability in probabilities:
            indices = np.argmax(cumulative >= probability, axis=0)
            quantiles.append(sorted_values[indices, np.arange(values.shape[1])])
        return tuple(quantiles)

    def _resolve_nuclei_groups(self, nuclei: Sequence[str]) -> list[tuple[str, np.ndarray]]:
        groups: list[tuple[str, np.ndarray]] = []
        missing: list[str] = []

        for name in nuclei:
            if name in self.isotope_to_index:
                groups.append((name, np.array([self.isotope_to_index[name]], dtype=np.int64)))
                continue

            if name.isalpha():
                matching = [index for isotope, index in self.isotope_to_index.items() if isotope.startswith(name)]
                if matching:
                    groups.append((name, np.array(sorted(matching), dtype=np.int64)))
                    continue

            missing.append(name)

        if missing:
            raise KeyError(f"Unknown isotopes requested: {missing}")
        return groups

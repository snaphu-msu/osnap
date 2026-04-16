"""Helpers for projecting full isotope networks onto a target basis."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from .isotopes import parse_isotope

DEFAULT_TARGET_NET_PATH = Path("ccsn_weak_r_cs_ba.sunet")
DEFAULT_REDUCED_NET_PATH = DEFAULT_TARGET_NET_PATH
_NETWORK_ALIASES = {
    "n": "nt1",
    "neut": "nt1",
    "p": "h1",
    "d": "h2",
    "t": "h3",
}


@dataclass(frozen=True)
class ReducedNetwork:
    """Normalized target-network isotope metadata."""

    path: Path
    isotopes: np.ndarray
    a: np.ndarray
    z: np.ndarray
    z_over_a: np.ndarray


@dataclass(frozen=True)
class ReductionRecipe:
    """Sparse mapping from a source isotope basis to the target basis."""

    source_isotopes: np.ndarray
    target_isotopes: np.ndarray
    matrix: csr_matrix
    nt1_index: int
    h1_index: int
    edge_source_fraction: np.ndarray
    missing_element_fraction: np.ndarray
    light_correction_fraction: np.ndarray


def load_reduced_network(path: Path | str = DEFAULT_REDUCED_NET_PATH) -> ReducedNetwork:
    """Parse a target network file into a normalized isotope list."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Reduced network file not found: {path}")

    text = path.read_text(encoding="utf-8").strip()
    labels = [_normalize_label(token) for token in _iter_network_tokens(text)]
    if not labels:
        raise ValueError(f"Reduced network file is empty: {path}")

    isotopes = np.array(labels)
    isotope_meta = [parse_isotope(name) for name in labels]
    a = np.array([entry.a for entry in isotope_meta], dtype=np.int16)
    z = np.array([entry.z for entry in isotope_meta], dtype=np.int16)
    z_over_a = (z.astype(np.float64) / a.astype(np.float64)).astype(np.float32)
    return ReducedNetwork(path=path, isotopes=isotopes, a=a, z=z, z_over_a=z_over_a)


def build_reduction_recipe(source_isotopes: Sequence[str], target: ReducedNetwork) -> ReductionRecipe:
    """Build a sparse, deterministic full-to-target isotope remap."""

    target_index = {name: index for index, name in enumerate(target.isotopes.tolist())}
    try:
        nt1_index = target_index["nt1"]
        h1_index = target_index["h1"]
    except KeyError as exc:
        raise ValueError("Reduced target network must contain both 'nt1' and 'h1'.") from exc

    target_by_symbol: dict[str, list[tuple[int, float, int]]] = defaultdict(list)
    for index, name in enumerate(target.isotopes.tolist()):
        if name == "nt1":
            continue
        isotope = parse_isotope(name)
        symbol = _symbol(name)
        target_by_symbol[symbol].append((isotope.a, isotope.z_over_a, index))

    for symbol in target_by_symbol:
        target_by_symbol[symbol].sort(key=lambda item: item[0])

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    edge_source_fraction = np.zeros(len(source_isotopes), dtype=np.float32)
    missing_element_fraction = np.zeros(len(source_isotopes), dtype=np.float32)
    light_correction_fraction = np.zeros(len(source_isotopes), dtype=np.float32)

    for row_index, name in enumerate(source_isotopes):
        if name in target_index:
            rows.append(row_index)
            cols.append(target_index[name])
            data.append(1.0)
            continue

        isotope = parse_isotope(name)
        symbol = _symbol(name)
        chain = target_by_symbol.get(symbol)
        if chain is None:
            h_weight = isotope.z_over_a
            n_weight = 1.0 - isotope.z_over_a
            if n_weight > 0.0:
                rows.append(row_index)
                cols.append(nt1_index)
                data.append(n_weight)
            if h_weight > 0.0:
                rows.append(row_index)
                cols.append(h1_index)
                data.append(h_weight)
            missing_element_fraction[row_index] = 1.0
            light_correction_fraction[row_index] = 1.0
            continue

        a_values = np.array([entry[0] for entry in chain], dtype=np.int16)
        z_over_a_values = np.array([entry[1] for entry in chain], dtype=np.float64)
        chain_indices = np.array([entry[2] for entry in chain], dtype=np.int64)

        if a_values.shape[0] == 1 or isotope.a < a_values[0] or isotope.a > a_values[-1]:
            nearest_slot = int(np.argmin(np.abs(z_over_a_values - isotope.z_over_a)))
            nearest_target_index = int(chain_indices[nearest_slot])
            nearest_target_ye = float(z_over_a_values[nearest_slot])
            weights = _edge_light_weights(isotope.z_over_a, nearest_target_ye)
            target_weight = weights["target"]
            nt1_weight = weights["nt1"]
            h1_weight = weights["h1"]
            rows.append(row_index)
            cols.append(nearest_target_index)
            data.append(target_weight)
            if nt1_weight > 0.0:
                rows.append(row_index)
                cols.append(nt1_index)
                data.append(nt1_weight)
            if h1_weight > 0.0:
                rows.append(row_index)
                cols.append(h1_index)
                data.append(h1_weight)
            edge_source_fraction[row_index] = 1.0
            light_correction_fraction[row_index] = nt1_weight + h1_weight
            continue

        insertion = int(np.searchsorted(a_values, isotope.a))
        lower_slot = insertion - 1
        upper_slot = insertion
        lower_target_ye = float(z_over_a_values[lower_slot])
        upper_target_ye = float(z_over_a_values[upper_slot])
        denominator = lower_target_ye - upper_target_ye
        if denominator <= 0.0:
            nearest_slot = int(np.argmin(np.abs(z_over_a_values - isotope.z_over_a)))
            rows.append(row_index)
            cols.append(int(chain_indices[nearest_slot]))
            data.append(1.0)
            edge_source_fraction[row_index] = 1.0
            continue

        lower_weight = float((isotope.z_over_a - upper_target_ye) / denominator)
        lower_weight = float(np.clip(lower_weight, 0.0, 1.0))
        upper_weight = 1.0 - lower_weight
        rows.extend((row_index, row_index))
        cols.extend((int(chain_indices[lower_slot]), int(chain_indices[upper_slot])))
        data.extend((lower_weight, upper_weight))

    matrix = csr_matrix(
        (np.asarray(data, dtype=np.float64), (np.asarray(rows), np.asarray(cols))),
        shape=(len(source_isotopes), target.isotopes.shape[0]),
    )
    return ReductionRecipe(
        source_isotopes=np.asarray(source_isotopes),
        target_isotopes=target.isotopes.copy(),
        matrix=matrix,
        nt1_index=nt1_index,
        h1_index=h1_index,
        edge_source_fraction=edge_source_fraction,
        missing_element_fraction=missing_element_fraction,
        light_correction_fraction=light_correction_fraction,
    )


def project_compositions(
    compositions: np.ndarray,
    source_z_over_a: np.ndarray,
    recipe: ReductionRecipe,
    target_z_over_a: np.ndarray,
    *,
    ye_tolerance: float = 1.0e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Project full compositions onto the target basis and repair tiny Ye drift."""

    compositions = np.asarray(compositions, dtype=np.float64)
    source_z_over_a = np.asarray(source_z_over_a, dtype=np.float64)
    target_z_over_a = np.asarray(target_z_over_a, dtype=np.float64)

    reduced = recipe.matrix.T.dot(compositions.T).T
    full_ye = compositions @ source_z_over_a
    projected_ye = reduced @ target_z_over_a
    ye_error_before = projected_ye - full_ye
    used_optimizer = np.zeros(compositions.shape[0], dtype=np.int8)

    for row_index, delta in enumerate(full_ye - projected_ye):
        if abs(delta) <= ye_tolerance:
            continue
        if delta > 0.0 and reduced[row_index, recipe.nt1_index] + 1.0e-12 >= delta:
            reduced[row_index, recipe.nt1_index] -= delta
            reduced[row_index, recipe.h1_index] += delta
            continue
        if delta < 0.0 and reduced[row_index, recipe.h1_index] + 1.0e-12 >= -delta:
            reduced[row_index, recipe.h1_index] += delta
            reduced[row_index, recipe.nt1_index] -= delta
            continue

        reduced[row_index] = _project_to_constraints(reduced[row_index], full_ye[row_index], target_z_over_a)
        used_optimizer[row_index] = 1

    reduced = np.clip(reduced, 0.0, None)
    reduced /= reduced.sum(axis=1, keepdims=True)

    ye_error_after = (reduced @ target_z_over_a) - full_ye
    qa = {
        "missing_element_mass": (compositions @ recipe.missing_element_fraction.astype(np.float64)).astype(np.float32),
        "edge_source_mass": (compositions @ recipe.edge_source_fraction.astype(np.float64)).astype(np.float32),
        "light_correction_mass": (compositions @ recipe.light_correction_fraction.astype(np.float64)).astype(np.float32),
        "ye_error_before": ye_error_before.astype(np.float32),
        "ye_error_after": ye_error_after.astype(np.float32),
        "used_optimizer": used_optimizer,
    }
    return reduced.astype(np.float32), qa


def _normalize_label(token: str) -> str:
    label = token.strip().strip("'").strip('"')
    return _NETWORK_ALIASES.get(label, label)


def _iter_network_tokens(text: str) -> list[str]:
    if "," in text:
        raw_tokens = text.split(",")
    else:
        raw_tokens = text.splitlines()
    return [token.strip() for token in raw_tokens if token.strip() and not token.lstrip().startswith("#")]


def _symbol(name: str) -> str:
    return "".join(char for char in name if char.isalpha())


def _edge_light_weights(source_ye: float, target_ye: float) -> dict[str, float]:
    if np.isclose(source_ye, target_ye, atol=1.0e-12):
        return {"target": 1.0, "nt1": 0.0, "h1": 0.0}
    if target_ye > source_ye:
        target_weight = float(np.clip(source_ye / target_ye, 0.0, 1.0))
        return {"target": target_weight, "nt1": 1.0 - target_weight, "h1": 0.0}

    target_weight = float(np.clip((1.0 - source_ye) / (1.0 - target_ye), 0.0, 1.0))
    return {"target": target_weight, "nt1": 0.0, "h1": 1.0 - target_weight}


def _project_to_constraints(values: np.ndarray, target_ye: float, z_over_a: np.ndarray) -> np.ndarray:
    guess = np.clip(values, 0.0, None)
    if guess.sum() <= 0.0:
        guess = np.full_like(values, 1.0 / values.shape[0])
    else:
        guess /= guess.sum()

    constraints = (
        {"type": "eq", "fun": lambda x: np.sum(x) - 1.0, "jac": lambda x: np.ones_like(x)},
        {"type": "eq", "fun": lambda x: np.dot(x, z_over_a) - target_ye, "jac": lambda x: z_over_a},
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
        raise RuntimeError(f"Failed to project reduced composition to Ye constraint: {result.message}")
    corrected = np.clip(result.x, 0.0, None)
    corrected /= corrected.sum()
    return corrected

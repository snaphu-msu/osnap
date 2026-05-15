"""Replace loaded progenitor compositions with Heger02 fitter predictions."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from composition_fitter.fitter import Heger02CompositionFitter, Prediction
from composition_fitter.heger02_dataset import DEFAULT_ARTIFACT_PATH as DEFAULT_HEGER02_COMPOSITION_ARTIFACT_PATH
from composition_fitter.isotopes import parse_isotope

DEFAULT_YE_COLUMNS = ("ye", "Ye", "Y_e", "cell Y_e")
DEFAULT_EXTRA_COMPOSITION_COLUMNS = ("prot", "co56", "cr60")
_SOURCE_LABEL_ALIASES = {
    "neut": "nt1",
    "neutrons": "nt1",
    "n": "nt1",
    "prot": "h1",
    "p": "h1",
}


@dataclass(frozen=True)
class CompositionReplacementResult:
    """Returned data for direct DataFrame composition replacement."""

    profiles: pd.DataFrame
    nuclear_network: list[str]
    diagnostics: pd.DataFrame


def replace_progenitor_composition(
    prog: dict,
    *,
    fitter: Heger02CompositionFitter | None = None,
    artifact_path: Path | str = DEFAULT_HEGER02_COMPOSITION_ARTIFACT_PATH,
    density_col: str = "density",
    temperature_col: str = "temp",
    ye_columns: Sequence[str] = DEFAULT_YE_COLUMNS,
    extra_composition_columns: Sequence[str] = DEFAULT_EXTRA_COMPOSITION_COLUMNS,
    clip: bool = True,
    inplace: bool = False,
) -> dict:
    """Replace a loaded Kepler or MESA progenitor composition with Heger02 predictions.

    The returned progenitor keeps all non-composition profile columns, removes the
    columns named by the original ``prog["nuclear_network"]``, appends the fitter
    isotope basis, and updates ``prog["nuclear_network"]`` to that fitter basis.
    STIR profile data is intentionally outside this function's scope.
    """

    if "profiles" not in prog:
        raise KeyError("prog is missing required key 'profiles'.")
    if "nuclear_network" not in prog:
        raise KeyError("prog is missing required key 'nuclear_network'.")

    target_prog = prog if inplace else dict(prog)
    original_network = [str(name) for name in prog["nuclear_network"]]
    result = replace_dataframe_composition(
        prog["profiles"],
        original_network,
        fitter=fitter,
        artifact_path=artifact_path,
        density_col=density_col,
        temperature_col=temperature_col,
        ye_columns=ye_columns,
        extra_composition_columns=extra_composition_columns,
        clip=clip,
    )

    target_prog["profiles"] = result.profiles
    target_prog["original_nuclear_network"] = original_network
    target_prog["nuclear_network"] = result.nuclear_network
    target_prog["heger02_composition_diagnostics"] = result.diagnostics
    return target_prog


def replace_dataframe_composition(
    profiles: pd.DataFrame,
    nuclear_network: Sequence[str],
    *,
    fitter: Heger02CompositionFitter | None = None,
    artifact_path: Path | str = DEFAULT_HEGER02_COMPOSITION_ARTIFACT_PATH,
    density_col: str = "density",
    temperature_col: str = "temp",
    ye_columns: Sequence[str] = DEFAULT_YE_COLUMNS,
    extra_composition_columns: Sequence[str] = DEFAULT_EXTRA_COMPOSITION_COLUMNS,
    clip: bool = True,
) -> CompositionReplacementResult:
    """Return a DataFrame with source composition columns replaced by Heger02 output."""

    if density_col not in profiles.columns:
        raise KeyError(f"profiles is missing required density column {density_col!r}.")
    if temperature_col not in profiles.columns:
        raise KeyError(f"profiles is missing required temperature column {temperature_col!r}.")

    fitter = Heger02CompositionFitter.load(artifact_path) if fitter is None else fitter
    source_network = [str(name) for name in nuclear_network]
    source_columns = [name for name in source_network if name in profiles.columns]
    extra_source_columns = [name for name in extra_composition_columns if name in profiles.columns]
    ye_source_columns = _ordered_unique([*source_columns, *extra_source_columns])
    target_ye = _target_ye_values(profiles, ye_source_columns, ye_columns)

    density = profiles[density_col].to_numpy(dtype=np.float64)
    temperature = profiles[temperature_col].to_numpy(dtype=np.float64)
    fitter_isotopes = [str(name) for name in fitter.isotopes.tolist()]

    predicted = np.empty((profiles.shape[0], len(fitter_isotopes)), dtype=np.float32)
    diagnostics: list[dict[str, object]] = []
    z_over_a = fitter.z_over_a.astype(np.float64, copy=False)

    for position, row_index in enumerate(profiles.index):
        row_density = float(density[position])
        row_temperature = float(temperature[position])
        row_ye = float(target_ye[position])
        if not np.isfinite(row_density) or row_density <= 0.0:
            raise ValueError(f"Invalid density for row {row_index}: {row_density!r}.")
        if not np.isfinite(row_temperature) or row_temperature <= 0.0:
            raise ValueError(f"Invalid temperature for row {row_index}: {row_temperature!r}.")
        if not np.isfinite(row_ye):
            raise ValueError(f"Could not determine a finite target Ye for row {row_index}.")

        try:
            prediction = fitter.predict(row_density, row_temperature, row_ye, clip=clip)
        except Exception as exc:
            raise ValueError(f"Failed to fit Heger02 composition for row {row_index}.") from exc

        predicted[position] = prediction.mass_fractions
        diagnostics.append(
            _diagnostic_row(
                row_index=row_index,
                density=row_density,
                temperature=row_temperature,
                requested_ye=row_ye,
                prediction=prediction,
                z_over_a=z_over_a,
            )
        )

    composition = pd.DataFrame(predicted, columns=fitter_isotopes, index=profiles.index)
    replaced_columns = _ordered_unique([*ye_source_columns, *fitter_isotopes])
    updated_profiles = profiles.drop(columns=replaced_columns, errors="ignore")
    updated_profiles = pd.concat([updated_profiles, composition], axis=1)
    diagnostics_frame = pd.DataFrame(diagnostics, index=profiles.index)
    return CompositionReplacementResult(
        profiles=updated_profiles,
        nuclear_network=fitter_isotopes,
        diagnostics=diagnostics_frame,
    )


def _target_ye_values(
    profiles: pd.DataFrame,
    source_columns: Sequence[str],
    ye_columns: Sequence[str],
) -> np.ndarray:
    direct = _direct_ye_values(profiles, ye_columns)
    missing = ~np.isfinite(direct)
    if not np.any(missing):
        return direct

    fallback = _composition_ye_values(profiles, source_columns)
    target = direct.copy()
    target[missing] = fallback[missing]
    unresolved = np.flatnonzero(~np.isfinite(target))
    if unresolved.size:
        row_index = profiles.index[int(unresolved[0])]
        raise ValueError(f"Could not determine a finite target Ye for row {row_index}.")
    return target


def _direct_ye_values(profiles: pd.DataFrame, ye_columns: Sequence[str]) -> np.ndarray:
    direct = np.full(profiles.shape[0], np.nan, dtype=np.float64)
    for column in ye_columns:
        if column not in profiles.columns:
            continue
        values = profiles[column].to_numpy(dtype=np.float64)
        use_column = ~np.isfinite(direct) & np.isfinite(values)
        direct[use_column] = values[use_column]
    return direct


def _composition_ye_values(profiles: pd.DataFrame, source_columns: Sequence[str]) -> np.ndarray:
    if not source_columns:
        return np.full(profiles.shape[0], np.nan, dtype=np.float64)

    values = profiles.loc[:, list(source_columns)].to_numpy(dtype=np.float64)
    values = np.where(np.isfinite(values), values, 0.0)
    mass_sum = values.sum(axis=1)
    weights = np.array([_z_over_a_for_source_label(label) for label in source_columns], dtype=np.float64)
    ye = values @ weights

    with np.errstate(invalid="ignore", divide="ignore"):
        ye = ye / mass_sum
    ye[mass_sum <= 0.0] = np.nan
    return ye


def _z_over_a_for_source_label(label: str) -> float:
    normalized = _SOURCE_LABEL_ALIASES.get(str(label).strip().lower(), str(label).strip().lower())
    if normalized == "fe":
        return 26.0 / 56.0
    return parse_isotope(normalized).z_over_a


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _diagnostic_row(
    *,
    row_index: object,
    density: float,
    temperature: float,
    requested_ye: float,
    prediction: Prediction,
    z_over_a: np.ndarray,
) -> dict[str, object]:
    mass_fractions = prediction.mass_fractions.astype(np.float64, copy=False)
    return {
        "row_index": row_index,
        "density": density,
        "temperature": temperature,
        "requested_ye": requested_ye,
        "query_ye": float(prediction.query[2]),
        "fitted_ye": float(mass_fractions @ z_over_a),
        "clipped": bool(prediction.clipped),
        "neighbor_count": int(prediction.neighbor_count),
        "effective_sample_size": float(prediction.effective_sample_size),
        "warnings": " | ".join(prediction.warnings),
    }

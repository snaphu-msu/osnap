"""Replace loaded progenitor compositions with Ye-repaired reduced abundances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from composition_fitter.reduced_composition import (
    DEFAULT_UNSTABLE_ISOTOPES,
    RepairedComposition,
    repair_reduced_abundances,
)

DEFAULT_YE_COLUMNS = ("ye", "Ye", "Y_e", "cell Y_e")
DEFAULT_DROP_COMPOSITION_COLUMNS = ("prot", "co56", "cr60")


@dataclass(frozen=True)
class ReducedCompositionReplacementResult:
    """Returned data for direct DataFrame reduced-composition replacement."""

    profiles: pd.DataFrame
    nuclear_network: list[str]
    diagnostics: pd.DataFrame
    repaired: RepairedComposition


def replace_progenitor_reduced_composition(
    prog: dict,
    *,
    ye_columns: Sequence[str] = DEFAULT_YE_COLUMNS,
    unstable_isotopes: Sequence[str] = DEFAULT_UNSTABLE_ISOTOPES,
    drop_composition_columns: Sequence[str] = DEFAULT_DROP_COMPOSITION_COLUMNS,
    inplace: bool = False,
) -> dict:
    """Replace a loaded progenitor composition with Ye-repaired reduced abundances."""

    if "profiles" not in prog:
        raise KeyError("prog is missing required key 'profiles'.")
    if "nuclear_network" not in prog:
        raise KeyError("prog is missing required key 'nuclear_network'.")

    target_prog = prog if inplace else dict(prog)
    original_network = [str(name) for name in prog["nuclear_network"]]
    result = replace_dataframe_reduced_composition(
        prog["profiles"],
        original_network,
        ye_columns=ye_columns,
        unstable_isotopes=unstable_isotopes,
        drop_composition_columns=drop_composition_columns,
    )

    target_prog["profiles"] = result.profiles
    target_prog["original_nuclear_network"] = original_network
    target_prog["nuclear_network"] = result.nuclear_network
    target_prog["reduced_composition_diagnostics"] = result.diagnostics
    return target_prog


def replace_dataframe_reduced_composition(
    profiles: pd.DataFrame,
    nuclear_network: Sequence[str],
    *,
    ye_columns: Sequence[str] = DEFAULT_YE_COLUMNS,
    unstable_isotopes: Sequence[str] = DEFAULT_UNSTABLE_ISOTOPES,
    drop_composition_columns: Sequence[str] = DEFAULT_DROP_COMPOSITION_COLUMNS,
) -> ReducedCompositionReplacementResult:
    """Return a DataFrame with source composition columns replaced by repaired output."""

    source_network = [str(name) for name in nuclear_network]
    source_columns = [name for name in source_network if name in profiles.columns]
    if not source_columns:
        raise ValueError("No nuclear_network columns were present in profiles.")

    target_ye = _direct_ye_values(profiles, ye_columns)
    missing = np.flatnonzero(~np.isfinite(target_ye))
    if missing.size:
        row_index = profiles.index[int(missing[0])]
        raise ValueError(f"Could not determine a finite direct Ye for row {row_index}.")

    abundances = profiles.loc[:, source_columns].to_numpy(dtype=np.float64)
    repaired = repair_reduced_abundances(
        source_columns,
        abundances,
        target_ye,
        unstable_isotopes=unstable_isotopes,
    )

    repaired_labels = [str(label) for label in repaired.labels.tolist()]
    composition = pd.DataFrame(repaired.mass_fractions, columns=repaired_labels, index=profiles.index)
    replaced_columns = _ordered_unique([*source_columns, *drop_composition_columns, *repaired_labels])
    updated_profiles = profiles.drop(columns=replaced_columns, errors="ignore")
    updated_profiles = pd.concat([updated_profiles, composition], axis=1)
    diagnostics = pd.DataFrame(repaired.diagnostics, index=profiles.index)
    diagnostics.insert(0, "target_ye", repaired.target_ye)

    return ReducedCompositionReplacementResult(
        profiles=updated_profiles,
        nuclear_network=repaired_labels,
        diagnostics=diagnostics,
        repaired=repaired,
    )


def _direct_ye_values(profiles: pd.DataFrame, ye_columns: Sequence[str]) -> np.ndarray:
    direct = np.full(profiles.shape[0], np.nan, dtype=np.float64)
    for column in ye_columns:
        if column not in profiles.columns:
            continue
        values = profiles[column].to_numpy(dtype=np.float64)
        use_column = ~np.isfinite(direct) & np.isfinite(values)
        direct[use_column] = values[use_column]
    return direct


def _ordered_unique(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered

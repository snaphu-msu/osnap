"""Read and normalize Sukhbold presupernova profile data for comparison plots."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_SUKHBOLD_PROFILE_PATH = Path("progenitors/sukhbold_2016/s25.0_presn")
SUKHBOLD_PREFIX_TOKEN_COUNT = 14


@dataclass(frozen=True)
class SukhboldProfile:
    """Parsed radius profile and composition network from a Sukhbold presn file."""

    path: Path
    enclosed_mass_g: np.ndarray
    radius_cm: np.ndarray
    log_radius_cm: np.ndarray
    density: np.ndarray
    temperature: np.ndarray
    ye: np.ndarray
    source_labels: np.ndarray
    selector_labels: np.ndarray
    source_abundances: np.ndarray


def _normalize_sukhbold_label(label: str) -> str:
    stripped = label.strip()
    if stripped == "neutrons":
        return "nt1"
    stripped = stripped.replace("'", "")
    return stripped.lower()


def _extract_abundance_labels(header_line: str) -> list[str]:
    if "NETWORK" not in header_line:
        raise ValueError("Sukhbold header is missing the NETWORK delimiter.")

    tail = header_line.split("NETWORK", 1)[1].strip()
    labels = tail.split()
    if not labels:
        raise ValueError("Sukhbold header did not contain any abundance labels.")
    return labels


def read_sukhbold_profile(path: Path | str = DEFAULT_SUKHBOLD_PROFILE_PATH) -> SukhboldProfile:
    """Read the Sukhbold `s25.0_presn` profile into aligned numpy arrays."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Sukhbold profile not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        next(handle)  # version line
        source_labels = _extract_abundance_labels(next(handle).rstrip("\n"))

        enclosed_mass_values: list[float] = []
        radius_values: list[float] = []
        density_values: list[float] = []
        temperature_values: list[float] = []
        ye_values: list[float] = []
        abundance_rows: list[list[float]] = []

        for raw_line in handle:
            tokens = raw_line.split()
            if not tokens or not tokens[0].rstrip(":").isdigit():
                continue
            if len(tokens) < SUKHBOLD_PREFIX_TOKEN_COUNT:
                raise ValueError(f"Malformed Sukhbold row in {path}: {raw_line.rstrip()}")

            abundance_tokens = tokens[SUKHBOLD_PREFIX_TOKEN_COUNT:]
            if len(abundance_tokens) < len(source_labels):
                abundance_tokens = abundance_tokens + ["---"] * (len(source_labels) - len(abundance_tokens))
            elif len(abundance_tokens) > len(source_labels):
                abundance_tokens = abundance_tokens[: len(source_labels)]

            enclosed_mass_g = float(tokens[1])
            radius_cm = float(tokens[2])
            density = float(tokens[4])
            temperature = float(tokens[5])
            ye = float(tokens[11])
            if radius_cm <= 0.0:
                raise ValueError(f"Encountered non-positive radius in {path}: {radius_cm}")
            if density <= 0.0:
                raise ValueError(f"Encountered non-positive density in {path}: {density}")
            if temperature <= 0.0:
                raise ValueError(f"Encountered non-positive temperature in {path}: {temperature}")

            abundances = [0.0 if token == "---" else float(token) for token in abundance_tokens]
            enclosed_mass_values.append(enclosed_mass_g)
            radius_values.append(radius_cm)
            density_values.append(density)
            temperature_values.append(temperature)
            ye_values.append(ye)
            abundance_rows.append(abundances)

    selector_labels = np.array([_normalize_sukhbold_label(label) for label in source_labels])
    enclosed_mass_g = np.asarray(enclosed_mass_values, dtype=np.float64)
    radius_cm = np.asarray(radius_values, dtype=np.float64)
    density = np.asarray(density_values, dtype=np.float64)
    temperature = np.asarray(temperature_values, dtype=np.float64)
    ye = np.asarray(ye_values, dtype=np.float64)
    source_abundances = np.asarray(abundance_rows, dtype=np.float64)

    return SukhboldProfile(
        path=path,
        enclosed_mass_g=enclosed_mass_g.astype(np.float32),
        radius_cm=radius_cm.astype(np.float32),
        log_radius_cm=np.log10(radius_cm).astype(np.float32),
        density=density.astype(np.float32),
        temperature=temperature.astype(np.float32),
        ye=ye.astype(np.float32),
        source_labels=np.array(source_labels),
        selector_labels=selector_labels,
        source_abundances=source_abundances.astype(np.float32),
    )

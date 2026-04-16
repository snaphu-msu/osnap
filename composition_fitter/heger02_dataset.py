"""Build and load cached fit artifacts for Heger02 presupernova models."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .isotopes import parse_isotope
from .reduced_network import (
    DEFAULT_TARGET_NET_PATH,
    build_reduction_recipe,
    load_reduced_network,
    project_compositions,
)

HEADER_OFFSET = 7
HEADER_WIDTH = 25
DEFAULT_SOURCE_DIR = Path("progenitors/heger02")
DEFAULT_FULL_ARTIFACT_PATH = Path("output/heger02_fitter/heger02_artifact.npz")
DEFAULT_TARGET_ARTIFACT_PATH = Path("output/heger02_fitter/heger02_ccsn_weak_r_cs_ba_artifact.npz")
DEFAULT_ARTIFACT_PATH = DEFAULT_TARGET_ARTIFACT_PATH
DEFAULT_REDUCED_ARTIFACT_PATH = DEFAULT_TARGET_ARTIFACT_PATH

REQUIRED_COLUMNS = (
    "cell density",
    "cell temperature",
    "cell Y_e",
    "stability",
)
COMPOSITION_START_INDEX = 12


@dataclass(frozen=True)
class PresnSchema:
    """Header information for a single `@presn` file."""

    path: Path
    columns: tuple[str, ...]
    isotopes: tuple[str, ...]
    zone_count: int


def split_header(line: str, start: int = HEADER_OFFSET, width: int = HEADER_WIDTH) -> list[str]:
    """Split the fixed-width header line into field names."""

    body = line[start:].rstrip("\n")
    return [body[index : index + width].strip() for index in range(0, len(body), width)]


def iter_presn_paths(source_dir: Path) -> list[Path]:
    """Return the `@presn` files in deterministic order."""

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    paths = sorted(source_dir.glob("*@presn"))
    if not paths:
        raise FileNotFoundError(f"No '*@presn' files found under: {source_dir}")
    return paths


def scan_presn_directory(source_dir: Path) -> list[PresnSchema]:
    """Scan headers and zone counts without building the dense artifact."""

    schemas: list[PresnSchema] = []
    for path in iter_presn_paths(source_dir):
        with path.open("r", encoding="utf-8") as handle:
            next(handle)  # version line
            columns = tuple(split_header(next(handle)))
            zone_count = sum(1 for line in handle if line.split() and line.split()[0].rstrip(":").isdigit())

        missing = [column for column in REQUIRED_COLUMNS if column not in columns]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        schemas.append(
            PresnSchema(
                path=path,
                columns=columns,
                isotopes=columns[COMPOSITION_START_INDEX:],
                zone_count=zone_count,
            )
        )
    return schemas


def build_union_isotopes(schemas: Iterable[PresnSchema]) -> list[str]:
    """Preserve first-seen isotope order across all progenitor files."""

    ordered = OrderedDict()
    for schema in schemas:
        for isotope in schema.isotopes:
            ordered.setdefault(isotope, None)
    return list(ordered)


def _parse_row_values(tokens: list[str], expected_count: int) -> list[str]:
    values = tokens[1:]
    if len(values) < expected_count:
        values = values + ["---"] * (expected_count - len(values))
    elif len(values) > expected_count:
        values = values[:expected_count]
    return values


def _compute_feature_stats(log_rho: np.ndarray, log_temp: np.ndarray, ye: np.ndarray) -> dict[str, np.ndarray]:
    feature_matrix = np.column_stack((log_rho, log_temp, ye)).astype(np.float32)
    feature_mean = feature_matrix.mean(axis=0).astype(np.float32)
    feature_std = feature_matrix.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std == 0.0, 1.0, feature_std).astype(np.float32)
    feature_min = feature_matrix.min(axis=0).astype(np.float32)
    feature_max = feature_matrix.max(axis=0).astype(np.float32)
    return {
        "feature_matrix": feature_matrix,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_min": feature_min,
        "feature_max": feature_max,
    }


def _build_artifact_arrays(schemas: list[PresnSchema]) -> dict[str, np.ndarray]:
    union_isotopes = build_union_isotopes(schemas)
    isotope_index = {isotope: index for index, isotope in enumerate(union_isotopes)}

    isotope_meta = [parse_isotope(name) for name in union_isotopes]
    a = np.array([entry.a for entry in isotope_meta], dtype=np.int16)
    z = np.array([entry.z for entry in isotope_meta], dtype=np.int16)
    z_over_a = (z.astype(np.float64) / a.astype(np.float64)).astype(np.float32)

    total_zones = sum(schema.zone_count for schema in schemas)
    compositions = np.zeros((total_zones, len(union_isotopes)), dtype=np.float32)
    log_rho = np.empty(total_zones, dtype=np.float32)
    log_temp = np.empty(total_zones, dtype=np.float32)
    ye = np.empty(total_zones, dtype=np.float32)
    raw_ye = np.empty(total_zones, dtype=np.float32)
    zone_number = np.empty(total_zones, dtype=np.int32)
    zone_label = np.empty(total_zones, dtype="<U32")
    progenitor_id = np.empty(total_zones, dtype=np.int16)
    stability = np.empty(total_zones, dtype="<U32")
    progenitor_names = np.array([schema.path.name for schema in schemas])
    zone_counts = np.array([schema.zone_count for schema in schemas], dtype=np.int32)
    zero_sum_fallback_count = 0

    row = 0
    for file_id, schema in enumerate(schemas):
        local_indices = [isotope_index[isotope] for isotope in schema.isotopes]
        density_index = schema.columns.index("cell density")
        temperature_index = schema.columns.index("cell temperature")
        ye_index = schema.columns.index("cell Y_e")
        stability_index = schema.columns.index("stability")

        with schema.path.open("r", encoding="utf-8") as handle:
            next(handle)
            next(handle)
            zone_counter = 0
            for line in handle:
                tokens = line.split()
                if not tokens:
                    continue
                if not tokens[0].rstrip(":").isdigit():
                    continue
                zone_counter += 1

                values = _parse_row_values(tokens, len(schema.columns))
                density = float(values[density_index])
                temperature = float(values[temperature_index])
                raw_file_ye = float(values[ye_index])

                local_composition = np.zeros(len(union_isotopes), dtype=np.float32)
                for local_slot, token in enumerate(values[COMPOSITION_START_INDEX:]):
                    if token == "---":
                        continue
                    value = float(token)
                    if value == 0.0:
                        continue
                    local_composition[local_indices[local_slot]] = value

                comp_sum = float(local_composition.sum())
                if comp_sum <= 0.0:
                    # Some zones have no explicit abundance table entries. Fall back to the
                    # minimal neutron/proton mixture implied by the file Ye so downstream
                    # queries still land on a valid simplex point.
                    try:
                        neutron_index = isotope_index["nt1"]
                        proton_index = isotope_index["h1"]
                    except KeyError as exc:
                        raise ValueError(
                            "Zero-sum composition encountered, but the isotope basis is missing "
                            "'nt1' or 'h1' for the fallback projection."
                        ) from exc
                    fallback_ye = float(np.clip(raw_file_ye, 0.0, 1.0))
                    local_composition[neutron_index] = 1.0 - fallback_ye
                    local_composition[proton_index] = fallback_ye
                    comp_sum = 1.0
                    zero_sum_fallback_count += 1
                local_composition /= comp_sum

                compositions[row] = local_composition
                log_rho[row] = np.log10(density)
                log_temp[row] = np.log10(temperature)
                ye[row] = float(local_composition @ z_over_a)
                raw_ye[row] = raw_file_ye
                label = tokens[0].rstrip(":")
                zone_label[row] = label
                zone_number[row] = zone_counter
                progenitor_id[row] = file_id
                stability[row] = values[stability_index]
                row += 1

    arrays = {
        "artifact_version": np.array([2], dtype=np.int16),
        "artifact_mode": np.array(["full"]),
        "compositions": compositions,
        "log_rho": log_rho,
        "log_temp": log_temp,
        "ye": ye,
        "raw_ye": raw_ye,
        "isotopes": np.array(union_isotopes),
        "a": a,
        "z": z,
        "z_over_a": z_over_a,
        "zone_number": zone_number,
        "zone_label": zone_label,
        "progenitor_id": progenitor_id,
        "progenitor_names": progenitor_names,
        "stability": stability,
        "zone_counts": zone_counts,
        "zero_sum_fallback_count": np.array([zero_sum_fallback_count], dtype=np.int32),
    }
    arrays.update(_compute_feature_stats(log_rho, log_temp, ye))
    return arrays


def _reduce_artifact_arrays(
    full_arrays: dict[str, np.ndarray],
    reduced_net_path: Path | str,
) -> dict[str, np.ndarray]:
    reduced_net = load_reduced_network(reduced_net_path)
    recipe = build_reduction_recipe(full_arrays["isotopes"].tolist(), reduced_net)
    reduced_compositions, qa = project_compositions(
        full_arrays["compositions"].astype(np.float64),
        full_arrays["z_over_a"].astype(np.float64),
        recipe,
        reduced_net.z_over_a.astype(np.float64),
    )
    reduced_ye = (reduced_compositions.astype(np.float64) @ reduced_net.z_over_a.astype(np.float64)).astype(np.float32)

    arrays = dict(full_arrays)
    arrays.update(
        {
            "artifact_version": np.array([2], dtype=np.int16),
            "artifact_mode": np.array(["reduced"]),
            "compositions": reduced_compositions.astype(np.float32),
            "ye": reduced_ye,
            "isotopes": reduced_net.isotopes.copy(),
            "a": reduced_net.a.copy(),
            "z": reduced_net.z.copy(),
            "z_over_a": reduced_net.z_over_a.copy(),
            "target_net_path": np.array([str(Path(reduced_net_path))]),
            "reduced_net_path": np.array([str(Path(reduced_net_path))]),
            "reduction_source_isotopes": full_arrays["isotopes"].copy(),
            "reduction_nonzero_count": np.array([recipe.matrix.nnz], dtype=np.int32),
            "projection_missing_element_mass": qa["missing_element_mass"],
            "projection_edge_source_mass": qa["edge_source_mass"],
            "projection_light_correction_mass": qa["light_correction_mass"],
            "projection_ye_error_before": qa["ye_error_before"],
            "projection_ye_error_after": qa["ye_error_after"],
            "projection_used_optimizer": qa["used_optimizer"],
        }
    )
    arrays.update(_compute_feature_stats(full_arrays["log_rho"], full_arrays["log_temp"], reduced_ye))
    return arrays


def build_artifact(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    artifact_path: Path | str | None = None,
    *,
    reduced_net_path: Path | str | None = None,
) -> Path:
    """Parse the raw Heger02 files and write a cached ``.npz`` artifact."""

    source_dir = Path(source_dir)
    if artifact_path is None:
        artifact_path = DEFAULT_REDUCED_ARTIFACT_PATH if reduced_net_path is not None else DEFAULT_FULL_ARTIFACT_PATH
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    schemas = scan_presn_directory(source_dir)
    arrays = _build_artifact_arrays(schemas)
    if reduced_net_path is not None:
        arrays = _reduce_artifact_arrays(arrays, reduced_net_path)
    np.savez_compressed(artifact_path, **arrays)
    return artifact_path


def build_reduced_artifact(
    source_dir: Path | str = DEFAULT_SOURCE_DIR,
    artifact_path: Path | str = DEFAULT_REDUCED_ARTIFACT_PATH,
    reduced_net_path: Path | str = DEFAULT_TARGET_NET_PATH,
) -> Path:
    """Build a target-basis artifact using the configured network file."""

    return build_artifact(source_dir, artifact_path, reduced_net_path=reduced_net_path)


def load_artifact(artifact_path: Path | str = DEFAULT_ARTIFACT_PATH) -> dict[str, np.ndarray]:
    """Load a previously built artifact into plain numpy arrays."""

    artifact_path = Path(artifact_path)
    with np.load(artifact_path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a cached Heger02 fitter artifact.")
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE_DIR,
        help="Directory containing the raw '*@presn' files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz artifact path.",
    )
    parser.add_argument(
        "--reduced-net",
        type=Path,
        default=None,
        help="Optional target network file; if provided, a target-basis artifact is built.",
    )
    args = parser.parse_args()

    artifact_path = build_artifact(args.source, args.output, reduced_net_path=args.reduced_net)
    arrays = load_artifact(artifact_path)
    print(f"Wrote artifact: {artifact_path}")
    print(
        "mode=",
        arrays["artifact_mode"][0],
        "zones=",
        arrays["feature_matrix"].shape[0],
        "isotopes=",
        arrays["isotopes"].shape[0],
        "rho_range=",
        (10 ** arrays["feature_min"][0], 10 ** arrays["feature_max"][0]),
        "temp_range=",
        (10 ** arrays["feature_min"][1], 10 ** arrays["feature_max"][1]),
        "ye_range=",
        (arrays["feature_min"][2], arrays["feature_max"][2]),
    )


if __name__ == "__main__":
    main()

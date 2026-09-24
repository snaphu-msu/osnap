# ODB: OSNAP database

`osnap.odb` is the **OSNAP database (ODB)**. It provides the common scientific
objects, units, and numerical operations used to exchange data between OSNAP
components, with an in-memory object model and optional HDF5 persistence. ODB is
independent of the legacy configuration system and simulation-code packages.
Import its public API directly from `osnap.odb`.

The model supports one-dimensional spherical Eulerian and Lagrangian grids,
independent tracer histories, composition, and grid-free products such as spectra
and diagnostic histories. Each snapshot can have its own grid, fields, and
species table. EOS evaluation, network execution, trajectory integration,
composition fitting, and explosion diagnostics belong in separate physics modules.

**Implementation status:** the core objects, units, numerical operations, and
selective HDF5 I/O have tests in [`tests/odb`](../../tests/odb/). Readers and
exporters for the seven native format families below are implemented, but this
checkout does not yet contain representative adapter fixtures or downstream-reader
smoke tests. Treat those implementations as requiring format-specific validation
before migrating production workflows. This README describes the current code,
including its restrictions, rather than claiming every design acceptance criterion
has been completed.

## Development setup

Run from the repository root with Python 3.10 or newer:

```sh
python -m pip install -r requirements-odb.txt
python -m pip install pytest
python -m pytest tests/odb -q
```

Runtime dependencies are NumPy, h5py, and the standard library. The legacy
[`requirements.txt`](../../requirements.txt) is not needed for this module. There
is no separate ODB CLI or installable package configuration; run examples
from the repository root, or put that root on `PYTHONPATH`.

## Construct, validate, and save a dataset

This example is self-contained. Later examples reuse `snapshot`, `data`, and `np`.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from osnap.odb import (
    Composition, Dataset, Field, RadialGrid, Series, Snapshot, Species,
)

# An excised model: neither the inner radius nor enclosed mass is zero.
faces = np.array([1e8, 2e8, 4e8])  # cm
density = np.array([2.0, 1.0])    # g/cm^3; shell averages
geometry = RadialGrid(faces)
mass_faces = 1e25 + np.r_[0.0, np.cumsum(density * geometry.volumes())]
grid = RadialGrid(faces, mass_faces=mass_faces, frame="lagrangian")

composition = Composition(
    Field(
        "mass_fraction", [[0.7, 0.3], [0.4, 0.6]], "1",
        dimensions=("cell", "species"), location="cell", sampling="average",
        description="Hydrogen-1 and helium-4 mass fractions",
    ),
    [Species.isotope(1, 1), Species.isotope(2, 4)],
    basis="mass_fraction", complete=True,
)
snapshot = Snapshot(
    [
        Field("density", density, "g/cm^3", dimensions=("cell",),
              location="cell", sampling="average", description="Shell-average density"),
        Field("temperature", [1e7, 2e7], "K", dimensions=("cell",),
              location="cell", sampling="average", description="Shell-average temperature"),
        Field("radial_velocity", [100.0, 200.0, 400.0], "km/s",
              dimensions=("face",), location="face", sampling="point",
              description="Radial velocity at shell boundaries"),
    ],
    grid=grid, compositions={"composition": composition}, time=0.0,
    source_id="synthetic-example",
)
data = Dataset(
    series={"hydrodynamics": Series([snapshot], time_reference="simulation_start")},
    metadata={"purpose": "README example"},
)
data.validate().raise_for_errors()

with TemporaryDirectory() as directory:
    path = data.write(Path(directory) / "model.osnap.h5")
    with Dataset.open(path) as restored:
        saved = restored.series["hydrodynamics"].snapshot(0)
        rho = saved.field("density").read(slice(0, 1))
        velocity = saved.field("radial_velocity", location="face").read(unit="km/s")
        np.testing.assert_allclose(rho, [2.0])
        np.testing.assert_allclose(velocity, [100.0, 200.0, 400.0])
    # Returned NumPy arrays remain usable after the file closes.
    assert rho.shape == (1,)
```

## Object model and conventions

| Object | Responsibility and main entry points |
| --- | --- |
| `Dataset` | Named `series` and `tracers`, metadata, provenance, import `report`; `open`, `write`, `from_native`, `export_native`, `validate`. |
| `Series` | Ordered snapshots and a `time_reference`; `snapshot(index)`, `interpolate(...)`, `shifted_time(...)`. |
| `Snapshot` | One time, optional radial grid, named axes, fields, compositions; `field`, `with_field`, `derive`, `remap`, `select_cells`, `interpolate_field`, `reduce`. |
| `RadialGrid` | `radius_faces`, `radius_cells`, optional `mass_faces`, and `frame`; `volumes()` returns a NumPy array in cm³. |
| `Field` | Numeric values, units, placement, axes, sampling, validity, description, source metadata; `read`, `read_validity`, `require_valid`. |
| `Axis` | One-dimensional numeric coordinates or string labels, with optional bin `bounds`; `read`. |
| `Composition` | Abundance `Field`, species definitions, convention, completeness, interpretation; abundance and elemental operations. |
| `Trajectory` | Persistent tracer ID, time axis, sampled fields/compositions, optional represented mass and scalar properties; `field`, `interpolate`. |
| `TracerSet` | Trajectories keyed by ID; `trajectory(id)`, `species_yields(...)`. |
| `UnitRegistry` | Unit definitions, dimensional checks, conversion factors, CGS representations. |
| `Adapter` / `Report` | Native-format translation contract / structured conversion and validation results. |

### Fields, axes, and ownership

For `N` shells, radial faces have length `N + 1` and cell coordinates have length
`N`. Faces increase strictly; native cell sampling radii are retained and must lie
within their shells. Missing cell coordinates are constructed as **radial
midpoints**, recorded in grid metadata. They are not volume centroids.
Enclosed-mass faces may start above zero and are nondecreasing.

Every `Field` has an explicit unit (including `"1"`), a description, ordered
`dimensions`, and an `axes` mapping from dimension names to coordinate references.
The default mapping uses the dimension names themselves. Cell and face fields
start with the `"cell"` or `"face"` dimension; other dimensions reference named
snapshot axes. Composition objects supply their own species dimension.

`location` is `"cell"`, `"face"`, `"sample"`, or `"global"`. Global fields are scalar;
use sample fields and named axes for arrays without a radial grid. `sampling` is
`"point"`, `"average"`, or `"integral"` and defaults to `"point"`. Set it deliberately:
numerical operations use it to decide which transformations are valid.

Fields are keyed by `(location, name)`. Cell and face versions of
`radial_velocity` can coexist; `snapshot.field("radial_velocity")` then raises
`DataError` until a location is supplied. A missing field raises `KeyError`.

An optional `valid` boolean mask has the field's full shape. Without a mask,
`read_validity()` returns all true, even if values contain NaNs. `read()` preserves
the raw values; `require_valid()` reads the entire field and rejects any invalid
or nonfinite entry. Numerical operations generally require all entries of each
input field to be valid, even if a particular output would use only part of it.

Construction copies caller arrays by default, and `read()` returns a detached
NumPy array. Transformations such as `with_field()` return new containers and may
share unchanged objects. Containers and metadata dictionaries are not deeply
immutable. Prefer replacement operations over mutating their internals. The
lower-level `Field.replaced(values, ...)` does **not** carry a validity mask unless
you pass `valid=...`; `Field.subset(slice(...))` does preserve it.

Use scalar `Field` objects for physical properties, for example
`Field("represented_mass", 1e25, "g")`. General metadata must contain finite,
JSON-compatible values. Use `source` for native names, locations, and import
transformations, and `provenance` for dataset/snapshot/trajectory operations.

### Units

`Field` and numeric `Axis` construction converts linear units to CGS immediately.
For example, a field constructed with `"km/s"` stores cm/s values and records the
conversion in `source`. Canonical unit strings use base symbols and powers, so
`"g/cm^3"` becomes `"g*cm^-3"`. Compare units through the registry, not by spelling.
`field.read(unit="km/s")` converts the returned array without changing the field.

The registry accepts registered symbols, multiplication, division, parentheses,
and integer powers using `^` or `**`. Multiplication must be explicit. Dimensions
are ordered `(mass, length, time, temperature, baryon_count)`; unknown symbols and
incompatible conversions raise `UnitError`.

```python
from osnap.odb import UnitRegistry

units = UnitRegistry()
assert units.factor("kg/m^3", "g/cm^3") == 1e-3
units.register("code_length", scale=1e8, dimensions=(0, 1, 0, 0, 0))
length = Field("length", 2.0, "code_length", registry=units)
assert length.read(unit="cm") == 2e8
```

Default symbols include CGS/SI length, mass, time, energy, pressure, and frequency
units; `K`, `GK`, `M_sun`, `R_sun`, `L_sun`, `rad`, `sr`, `k_B`, and `baryon`.
See [`units.py`](units.py) for the exact symbols and adopted constants. Create a
separate registry for additional units and pass it to the relevant objects.
Definitions and their version are stored in HDF5; conflicting definitions cannot
be merged during writing.

The registry provides multiplicative conversions only. It does not track NumPy
arithmetic or perform temperature offsets, frequency/wavelength transformations,
or spectral-density conversions. Baryon count is a separate formal dimension;
`k_B/baryon` cannot be converted to entropy per gram without an explicit physical
operation. Magnitudes use `unit="mag"` and require
`metadata={"magnitude_system": "AB"}` (or the appropriate system); they are not
converted to flux. Native logarithmic columns are decoded by adapters, not by the
unit parser. Arrays needing no scale conversion retain their numeric dtype;
conversions with a factor other than one promote to float64. Remapping and
interpolation operate in float64.

### Composition

Abundance arrays end in a `"species"` dimension. Each `Species` has a stable ID,
source label, kind (`"isotope"`, `"element"`, or `"group"`), and applicable `Z`/`A`.
Groups need descriptions. For example, `Species.parse("fe", groups=("fe",))`
retains a source-defined group; it is not `fe56`.

`basis` is `"mass_fraction"` or `"number_per_baryon"`.
`mass_fractions()` returns a `Field`; for isotopic number abundances it uses
`X_i = A_i Y_i`. Source values, incomplete networks, and inconsistent totals are
preserved. `complete=True` is a declaration, not a normalization operation.

`normalize()` explicitly returns a new mass-fraction `Composition` with each row
normalized. It requires valid nonnegative values with positive row totals and
retains the original completeness flag. `electron_fraction()` requires a
complete isotopic composition and returns a separate
`composition_electron_fraction` field using `sum(X_i Z_i/A_i)`. It never overwrites
source `electron_fraction` data. Scientific validation reports meaningful
disagreements and normalization problems.

`elemental_sums()` returns a composition of elemental totals when species atomic
numbers are known. Keep overlapping element totals, isotopes, and residual groups
in separate named tables and declare `interpretation` (`"exclusive"`,
`"elemental_totals"`, or `"residual"`). Do not sum these tables together as if they
represented independent material.

### Time, tracers, and nonspatial data

Snapshot times are seconds, relative to their series' declared `time_reference`.
A single static snapshot may use `time=None`. Multi-snapshot validation and time
interpolation require known, strictly increasing times. Keep restart branches
separate. Shared adapter helpers split repeated/decreasing output times into
separate named branches rather than dropping rows.

Use `series.shifted_time(offset, time_reference="core_bounce")` to apply an
explicit offset in seconds before comparing different time references. A grid's
Eulerian/Lagrangian designation does not imply fixed resolution or persistent
cell identities.

```python
from osnap.odb import Axis, Trajectory, TracerSet

tracer = Trajectory(
    "tracer-42", [0.0, 1.0, 3.0],
    [Field("temperature", [1.0, 3.0, 7.0], "GK",
           dimensions=("time",), location="sample")],
    represented_mass=Field("represented_mass", 1e25, "g"),
    time_reference="explosion",
)
tracers = TracerSet([tracer])
sampled = tracers.trajectory("tracer-42").interpolate([0.5, 2.0], fields=["temperature"])
np.testing.assert_allclose(sampled.field("temperature").read(unit="GK"), [2.0, 5.0])

spectrum = Snapshot(
    [Field("luminosity_density", [1e38, 2e38], "erg/s/Angstrom",
           dimensions=("wavelength",), location="sample")],
    axes={"wavelength": Axis("wavelength", [4000.0, 5000.0], "Angstrom")},
    time=86400.0,
)
products = Dataset(
    series={"spectra": Series([spectrum], time_reference="explosion")},
    tracers={"ejecta": tracers},
)
products.validate().raise_for_errors()
```

Tracer IDs are integers or strings; their type is preserved and they are unrelated
to row order or enclosed mass. Each trajectory has an independent time axis.
Trajectory compositions use `("time", "species")` dimensions. Interpolation
returns only the requested field/composition names and forbids extrapolation.

`TracerSet.species_yields(composition="composition", sample=-1)` sums represented
mass times species mass fraction at the selected sample and returns scalar mass
fields by species ID. Every trajectory needs a represented mass. Callers must
ensure compatible species coverage and sample meaning across trajectories; the
current method accumulates species present in each table and does not enforce a
common network or a common physical time for the final samples.

Grid-free spectra use named axes, as above. Scalar diagnostic/light-curve histories
can use one snapshot per time. There is no general grid-free series interpolation
method at present.

## Derived fields and numerical operations

`Snapshot.derive()` returns a `Field` without modifying the snapshot or caching
results. Attach a materialized result with `snapshot.with_field(result)` when it
should be stored. Built-in derivations are:

| Name | Inputs and interpretation |
| --- | --- |
| `shell_width`, `face_area`, `cell_volume` | Spherical geometry from radial faces. Volumes use `4π/3 (r_outer³ − r_inner³)`. |
| `cell_mass` | Explicit `source="mass_faces"` or `source="density"`. Density must be a shell average, unless `assume_cell_average=True` is explicitly supplied. |
| `enclosed_mass` | Same mass-source choice plus explicit `inner_mass` in grams; returns a face field. |
| `composition_electron_fraction` | `composition="composition"` selects the table; requires a complete isotope network. |
| `species_mass` | Selected composition and explicit mass source; returns a `("cell", "species")` mass field. |

```python
volume = snapshot.derive("cell_volume")
mass = snapshot.derive("cell_mass", source="density")
mean_temperature = snapshot.reduce("temperature", operation="average", weights=mass)
assert mean_temperature.unit == "K"
enclosed = snapshot.derive("enclosed_mass", source="density", inner_mass=1e25)
np.testing.assert_allclose(enclosed.read(), snapshot.grid.mass_faces.read())
```

`reduce()` currently accepts one-dimensional fields, with `operation="sum"` or
`"average"`. Averages require explicit `Field` weights. Weighted sums multiply
the field and weight units; weighted averages retain the field unit. To attach a
derived `species_mass` field as an ordinary snapshot field, also provide a named
species `Axis`; ordinary fields do not inherit a composition's species axis.

### Centering and interpolation

`snapshot.interpolate_field(name, location=..., target_location=..., boundary=...)`
linearly interpolates **point-sampled** fields between cells and faces in radius.
Cell-to-face conversion usually requires boundary values `(inner_value,
outer_value)` in the field's stored units, or an explicit `"constant"`/`"linear"`
boundary rule. The default forbids extrapolation. Average and integral fields
cannot be relocated with this method.

### Conservative remapping

```python
remapped = snapshot.remap(
    [1e8, 1.5e8, 2e8, 3e8, 4e8],  # target faces in cm
    coordinate="radius", mass_source="density",
    field_policies={
        "temperature": "mass_weighted",
        ("face", "radial_velocity"): "point",
    },
)
np.testing.assert_allclose(
    remapped.derive("cell_mass", source="density").read().sum(),
    mass.read().sum(), rtol=1e-12,
)
```

Remapping transfers first-order, piecewise-constant overlaps. Radius remapping
uses spherical volume coordinate; mass remapping uses enclosed-mass intervals
and transfers shell volumes, reconstructing radii from the original inner radius.
The latter assumes constant specific volume within each source shell.

`coordinate="radius"` takes target boundaries in cm and requires an explicit
`mass_source` (`"density"` or `"mass_faces"`). `coordinate="mass"` takes boundaries
in grams, requires `mass_source="mass_faces"`, and requires strictly positive
source shell masses. Target endpoints must match the source domain. For
truncation, use `select_cells(start, stop, mass_source=...)` first; it reports
excluded mass in provenance and uses a half-open cell range.

Mass and represented species masses are conserved, with density and abundances
reconstructed afterward. Species tables remain unchanged. Zero-mass target cells
have invalid NaN abundances. Required missing or negative mass/abundance data
cause failure; incomplete abundance totals are not repaired. Momentum and energy
conservation are not guaranteed.

Density and all composition tables are carried automatically. **Other fields are
retained only when listed in `field_policies`**; omitted fields are recorded in
provenance. Policy keys are field names or `(location, name)` pairs:

| Policy | Supported input |
| --- | --- |
| `mass_weighted` | Cell averages, weighted by the selected mass source. |
| `volume_weighted` | Cell averages, weighted by shell volume. |
| `integral` | Cell integrals, transferred by overlap fraction. |
| `point` | Point-sampled cell or face fields; linear interpolation in the selected coordinate, without extrapolation. |
| `copy` | Global scalars, unchanged during spatial remapping. |

Do not give density a policy: it is reconstructed from transferred mass. Point
interpolation can fail near the endpoints when new cell sampling positions lie
outside the source sampling range, even when shell boundaries cover the same
domain. Face interpolation has no mass-conservation guarantee.

For time interpolation, call:

```python
# Here time=0 recovers the sole snapshot after aligning it to the target grid.
aligned = data.series["hydrodynamics"].interpolate(
    0.0, target_grid=remapped.grid, coordinate="radius", mass_source="density",
    field_policies={"temperature": "mass_weighted"},
    time_reference="simulation_start",
)
np.testing.assert_allclose(aligned.field("density").read(), remapped.field("density").read())
```

For a time between outputs, the bracketing snapshots are remapped first, then
cell masses and species masses are interpolated before density and abundances
are reconstructed. Composition tables, species definitions, and abundance
conventions must match between brackets. Radius alignment requires a common
radial domain; mass alignment requires a common enclosed-mass domain and can
interpolate moving radial boundaries. A supplied target grid does not permit
domain extrapolation. Weighted fields interpolate their extensive weighted
contributions; global scalars copied spatially are interpolated in time.

## HDF5 storage and lifecycle

The current schema is **`1.0`**, with writer version **`1.0.0`**. The reader rejects
other schema versions, unsupported required features, and unknown unit-definition
versions. [`storage.py`](storage.py) defines the authoritative serialization.

```text
/                                      attrs: format, schema_version, writer_version,
                                              dataset_id, created_utc
  metadata_json
  report_json
  units/                               attr: version
    definitions_json
  species_tables/<table_id>/species_json
  provenance/<record_id>                JSON dataset per record
  series/<name>/
    metadata_json                      time reference and series metadata
    index                              JSON records; attr: time_unit="s"
    snapshots/<storage_id>/
      metadata_json                    time, source identity, metadata, provenance
      grid/                            optional; metadata_json and coordinate fields
      axes/<name>/                     metadata_json, values, optional bounds field
      fields/<location>/<name>/        metadata_json, values, optional valid
      compositions/<name>/             metadata_json and abundances field
  tracers/<set_name>/
    metadata_json
    index                              JSON tracer ID / storage ID mapping
    trajectories/<storage_id>/
      metadata_json
      time/                            axis representation
      fields/<location>/<name>/        field representation
      compositions/<name>/
      properties/<name>/               scalar field representation
      represented_mass/               optional scalar field representation
```

Grid coordinate fields are named `radius_faces`, `radius_cells`, and optionally
`mass_faces`. Composition metadata references a deduplicated species table.
Storage IDs are ordinal strings, independent of floating-point time or tracer ID
formatting. Field metadata includes name, units, dimensions, axis references,
location, sampling, description, source, and general metadata. Common discovery
properties are also HDF5 attributes. Dtype and shape come from the values dataset.

Structured metadata uses UTF-8 JSON; no pickle or executable derivations are
stored. Unknown keys inside the user metadata dictionaries are retained. This
does not imply preservation of arbitrary foreign HDF5 groups or attributes.
Nonempty, nonscalar **Field** arrays and masks use chunking, gzip level 4, and
shuffle. Scalar and empty fields are uncompressed; `Axis.values` currently uses
ordinary datasets without those compression settings.

`Dataset.open()` reads indexes and metadata without reading field arrays.
Retrieving a snapshot creates field proxies; `Field.read(selection)` passes the
selection directly to h5py. Use slices (and h5py-supported indexing for disk-backed
fields) to avoid full reads. Scientific operations and validation have different
loading behavior: structural validation reads coordinates and tracer times;
scientific validation also reads field and composition arrays.

Keep `Dataset.open()` inside a `with` block, or call `close()` explicitly. Closing
invalidates disk-backed field reads and lazy snapshot/trajectory access with
`ClosedDatasetError`. Previously returned NumPy arrays remain valid; reading a
field once does not materialize or detach the field object itself.

`write(path, overwrite=False)` validates structure, writes a temporary sibling
file, reopens and validates it, and publishes the completed file atomically.
Existing targets require `overwrite=True`. Failed writes clean up the temporary
file. Source files must remain open while copying disk-backed data. This
publication guarantee applies to OSNAP HDF5 writes; native multi-file exports do
not provide the same transaction. Append-in-place, concurrent writers, and live
simulation monitoring are not implemented.

## Native adapters

`Dataset.from_native(source, format=..., **options)` returns a **Dataset**; inspect
`dataset.report` for transformations, assumptions, unmapped fields, information
loss, and issues. `Dataset.export_native(...)` returns a **Report**, not a dataset.
Reports can be converted to dictionaries with `to_dict()`.

Explicit format selection is authoritative. `format="auto"` must find exactly one
match; FLASH and STIR share a file family and can be ambiguous. Native reads are
eager; selective reading is provided by the OSNAP HDF5 format. Supported format
identifiers are lowercase in the table below (explicit dispatch is case-insensitive).

| Format / source | Implemented read paths and significant options | Implemented export |
| --- | --- | --- |
| [`flash`](adapters/flash.py) | FLASH HDF5 with per-variable block arrays, bounding boxes, node types, and 1D spherical geometry. Leaf cells are sorted and mesh continuity checked. `geometry`, `guard_cells`, `field_map`, `species_names`, `complete_composition`, `time_reference`. | ASCII progenitor profile for the supported supernova initialization layout; `columns` selects native labels and canonical cell fields. |
| [`stir`](adapters/flash.py) | FLASH-based `kind="profile"`; `kind="history"` requires a units-bearing `field_map`, with explicit `columns` when needed. | Same ASCII profile writer as FLASH. |
| [`mesa`](adapters/stellar.py) | Numbered-header text `kind="profile"`/`"history"`, and supported saved-model text (`kind="model"`). Reversed zones and logarithmic columns are converted explicitly. Profiles require inner boundaries or `full_star=True`. | `.mod` from a compatible `template`, matching isotope network, required fields, and explicit physical `header_values`. Structural headers are recomputed; previous-generation model data is dropped and reported. |
| [`kepler`](adapters/stellar.py) | Presupernova text with `NETWORK` header, or explicit `columns`/`species_names`. Requires inner boundaries or `full_star=True`. Ambiguous `fe` remains a group; missing values are masked. | None. |
| [`skynet`](adapters/skynet.py) | HDF5 `A`, `Z`, `Y(time, species)` and thermodynamic histories. `dataset_names` resolves aliases; `temperature_unit` is required if absent from the source. Optional `tracer_ids` and `represented_masses` (grams), one per file. | OSNAP driver directory: `trajectory.dat`, `initial_abundances.dat`, `manifest.json`. Time in seconds, temperature in GK, density in CGS, ordered isotope number abundances. |
| [`snec`](adapters/radiation.py) | `kind="model"` structure plus optional `composition_path`; `kind="profiles"` aligned `.xg` histories; `kind="history"` diagnostics; `kind="magnitudes"` requires columns and magnitude system. | Directory containing `model.short` and `composition.iso.dat`; requires `coordinate_location="outer_face"`. |
| [`tardis`](adapters/radiation.py) | `kind="model"` simple ASCII density/velocity and optional `abundance_path`; explicit handling of the first boundary row. `kind="spectrum"` needs `spectral_unit`. | Directory containing `density.dat` and `abundances.dat`; requires expansion `epoch` in seconds and compatible face velocities, or explicit `homologous=True` to construct them from radius. |

These are bounded file layouts, not support for every release, binary restart,
or variant. Adapter `mappings` and `read`/`write` implementations are the current
mapping references. In particular:

- FLASH assumes no guard cells unless `guard_cells` is supplied and checks block
  sizes against available parameters. It removes covered parents without
  resampling. Native `ener` maps to `specific_energy` with solver-defined energy
  terms, not automatically to internal energy. The default FLASH/STIR export
  requires cell fields `temperature`, `density`, `radial_velocity`,
  `electron_fraction`, and `sum_y`; none are synthesized.
- For outer-face stellar tables, supply `inner_radius` (cm) and `inner_mass` (g),
  or deliberately select `full_star=True` to set both to zero. MESA's missing
  inner face quantities can be supplied through `inner_fields`; otherwise those
  entries are invalid. MESA export `header_values` uses **native header units**,
  including years for `star_age`. Required headers depend on the template.
- SkyNet imports retain `number_per_baryon` abundances. Driver export selects one
  `Trajectory`, needs at least two samples, positive temperature/density, a
  complete normalized initial isotope composition, and consistent initial Ye.
  Only the initial composition sample is exported. The bundle is an OSNAP driver
  convention; it does not run SkyNet or represent a universal SkyNet input format.
- SNEC structure import and export require `coordinate_location="outer_face"` for
  the implemented progenitor-table convention. Export requires enclosed-mass
  faces plus cell temperature, density, velocity, Ye, and angular velocity. The
  `.xg` reader retains face fields and removes the outer ghost entry from cell
  fields.
- TARDIS read defaults to `abundance_format="labelled"`; export defaults to
  `"indexed"`. Pass the matching choice when reading an export. Current exports
  require elemental abundances within atomic numbers 1–30; isotope aggregation
  is reported as loss of isotopic detail. The density boundary row is not a cell.

Examples below require real native files; paths are illustrative:

```python
native = Dataset.from_native(
    "profile.data", format="mesa", kind="profile", full_star=True,
)
print(native.report.to_dict())
native.write("progenitor.osnap.h5")

ejecta = Dataset.from_native(
    "density.dat", format="tardis", abundance_path="abundances.dat",
    abundance_format="indexed",
)
report = ejecta.export_native(
    "tardis-input", format="tardis", selection=("ejecta", 0),
    epoch=ejecta.series["ejecta"].snapshot(0).time,
)
print(report.to_dict())
```

Snapshot exports accept a `(series_name, index)` pair or an actual `Snapshot`;
SkyNet export requires a `Trajectory` object. Export destinations refuse existing
targets unless `overwrite=True`. Exporters reject missing required fields or
incompatible compositions; prepare explicit centering conversions, remapping, or
normalization before exporting. Review `report.information_loss` and
`report.assumptions`, not just errors.

## Validation, extension, and maintenance

`Dataset.validate(level="structure")` checks dimensions, grids, time ordering,
identities, and represented masses. `level="scientific"` additionally reports
nonfinite/unphysical field values, composition totals, supplied versus derived Ye,
and disagreements between supplied and density-derived shell masses. It does not
repair data or establish full physical consistency.

Validation returns a `Report`. `report.ok` means there are no **error** issues;
warnings may still be present. `report.raise_for_errors()` raises `DataError` for
errors only. Inspect `report.issues` for warnings. Unit failures use `UnitError`,
unsupported formats/schemas use `FormatError`, and closed-file reads use
`ClosedDatasetError`.

### Registering a derived field

Declare each dependency's name, unit, and location; the callable receives detached
arrays in those units. Use `Dependency(..., derived=True)` for another registered
derivation instead of a stored field. Missing inputs and dependency cycles fail
explicitly. Registration is process-local and duplicate names are rejected.

```python
from osnap.odb import Dependency, register_derived

# Make the centering choice explicit before computing a cell quantity.
cell_velocity = snapshot.interpolate_field(
    "radial_velocity", location="face", target_location="cell",
)
with_velocity = snapshot.with_field(cell_velocity)

def radial_kinetic_energy(snapshot, inputs):
    return 0.5 * inputs["radial_velocity"] ** 2

register_derived(
    "specific_radial_kinetic_energy", radial_kinetic_energy,
    dependencies=[Dependency("radial_velocity", "cm/s", "cell")],
    unit="erg/g", location="cell", dimensions=("cell",), sampling="point",
    description="Specific kinetic energy of the radial velocity component", version="1",
)
energy = with_velocity.derive("specific_radial_kinetic_energy")
materialized = with_velocity.with_field(energy)
assert materialized.field("specific_radial_kinetic_energy").shape == (2,)
```

Optional `derive(..., **parameters)` are passed to the callable as keyword
arguments and recorded in its source metadata. Derived Python functions are never
serialized; materialized results retain their definition name, version, and
parameters. Keep physics requiring an EOS, network, or other external package
outside this module.

### Adding an adapter or format variant

1. Implement `Adapter.detect(source)`, `read(source, **options) -> Dataset`, and,
   when supported, `write(target, selection, **options) -> Report`. Declare the
   format identifier, file family, mappings, units, and placement assumptions.
2. Register an instance in [`adapters.ADAPTERS`](adapters/__init__.py). Keep detection
   conservative; explicit selection must remain usable when families overlap.
3. Record unit/log transformations, assumptions, unmapped fields, and information
   loss. Preserve unknown composition groups and source values. Fail when units,
   placement, or required metadata cannot be determined from documented inputs.
4. Add a small representative fixture and mapping table for each supported
   variant. Test native placement, logarithmic decoding, ordering, excision,
   composition identity, masks, and reports. Validate exports with an independent
   parser or the intended downstream reader; a round trip through the same
   adapter is insufficient evidence of native compatibility.

The existing implementation is organized as follows:

| File | Responsibility |
| --- | --- |
| [`units.py`](units.py), [`fields.py`](fields.py) | Unit grammar, array ownership, masks, axes, disk-backed array access. |
| [`composition.py`](composition.py) | Species identity and explicit abundance operations. |
| [`model.py`](model.py) | Public containers, grid conventions, reports, I/O dispatch. |
| [`derived.py`](derived.py), [`numerics.py`](numerics.py) | Derived registry, reductions via containers, interpolation, conservative overlaps. |
| [`storage.py`](storage.py), [`validation.py`](validation.py) | Versioned HDF5 encoding, lazy readers, atomic publication, validation. |
| [`adapters/`](adapters/) | Native-format readers, writers, and shared mapping helpers. |

Current core tests cover units, ownership/masks, composition preservation, field
placement, tracer histories, derived dependencies, remapping in both coordinates,
time interpolation, and HDF5 round trips. Numerical conservation tests use a
relative tolerance of `1e-12` for nonzero mass/species totals. Instrumented storage
tests verify that a slice read does not load unrelated arrays and that file closure
invalidates disk access. A subprocess test checks independence from legacy imports.

Outstanding acceptance work includes representative fixtures and independent
export validation for all seven adapters, native-reader smoke tests (especially
MESA saved models), and broader format-variant coverage. Multidimensional meshes,
native AMR hierarchy preservation, automatic network conversion, automatic
composition repair, and migration of legacy workflows remain outside ODB.

"""Algebraic derived fields and a dependency-checked extension registry."""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
import numpy as np

from .errors import DataError
from .fields import Field


@dataclass(frozen=True)
class Dependency:
    name: str
    unit: str
    location: str
    derived: bool = False


@dataclass(frozen=True)
class Definition:
    function: object
    dependencies: tuple
    unit: str
    location: str
    dimensions: tuple
    sampling: str
    description: str
    version: str


_CUSTOM = {}
_STACK = ContextVar("osnap_derivations", default=())
_BUILTINS = {"cell_volume", "shell_width", "face_area", "cell_mass", "enclosed_mass",
             "composition_electron_fraction", "species_mass"}


def register_derived(name, function, *, dependencies, unit, location, dimensions,
                     sampling="point", description, version="1"):
    """Register function(snapshot, dependencies, **parameters) -> numeric array.

    Dependencies are returned as detached arrays in the declared units. Each
    must explicitly select stored or derived data and a field location.
    """
    if name in _CUSTOM or name in _BUILTINS:
        raise DataError(f"Derived field {name!r} is already registered")
    if not callable(function) or not all(isinstance(d, Dependency) for d in dependencies):
        raise DataError("Supply a callable and explicit Dependency objects")
    if len({d.name for d in dependencies}) != len(dependencies):
        raise DataError("Dependency names must be unique")
    _CUSTOM[name] = Definition(function, tuple(dependencies), unit, location,
                               tuple(dimensions), sampling, description, str(version))


def derive(snapshot, name, **parameters):
    stack = _STACK.get()
    if name in stack:
        raise DataError("Derived-field dependency cycle: " + " -> ".join((*stack, name)))
    token = _STACK.set((*stack, name))
    try:
        if name in _CUSTOM:
            definition = _CUSTOM[name]
            inputs = {}
            for dep in definition.dependencies:
                f = derive(snapshot, dep.name) if dep.derived else snapshot.field(dep.name, location=dep.location)
                if f.location != dep.location:
                    raise DataError(f"Dependency {dep.name} has wrong centering")
                f.require_valid()
                inputs[dep.name] = f.read(unit=dep.unit)
            values = definition.function(snapshot, inputs, **parameters)
            registry = snapshot.grid.registry if snapshot.grid else next(iter(snapshot.fields.values())).registry
            field = Field(name, values, definition.unit, dimensions=definition.dimensions,
                          location=definition.location, sampling=definition.sampling,
                          description=definition.description, registry=registry,
                          source={"derivation": name, "version": definition.version,
                                  "parameters": parameters, "dependencies": [d.name for d in definition.dependencies]})
            Snapshot = type(snapshot)
            sizes = {k: a.shape[0] for k, a in snapshot.axes.items()}
            if snapshot.grid:
                sizes.update(cell=snapshot.grid.ncells, face=snapshot.grid.ncells+1)
            Snapshot._validate_field(field, sizes)
            return field
        return _builtin(snapshot, name, **parameters)
    finally:
        _STACK.reset(token)


def _builtin(s, name, **p):
    if name == "composition_electron_fraction":
        return s.compositions[p.get("composition", "composition")].electron_fraction()
    if name not in _BUILTINS:
        raise KeyError(f"Unknown derived field {name!r}")
    if s.grid is None:
        raise DataError(f"{name} requires a radial grid")
    g = s.grid
    g.validate()
    source = {"derivation": name, "version": "1", "parameters": p}

    def result(values, unit, location="cell", sampling="integral", dims=None):
        return Field(name, values, unit, location=location, dimensions=dims or (location,),
                     sampling=sampling, source=source, registry=g.registry)

    if name == "cell_volume":
        return result(g.volumes(), "cm^3")
    if name == "shell_width":
        return result(np.diff(g.radius_faces.read()), "cm")
    if name == "face_area":
        return result(4*np.pi*g.radius_faces.read()**2, "cm^2", "face")
    if name == "cell_mass":
        which = p.get("source")
        if which == "mass_faces":
            if g.mass_faces is None:
                raise DataError("No enclosed-mass boundaries are available")
            values = np.diff(g.mass_faces.require_valid())
        elif which == "density":
            rho = s.field("density", location="cell")
            if rho.sampling != "average" and not p.get("assume_cell_average", False):
                raise DataError("Density is not a shell average; explicitly set assume_cell_average=True")
            rho.registry.factor(rho.unit, "g/cm^3")
            values = rho.require_valid() * g.volumes()
        else:
            raise DataError("Specify mass source='density' or 'mass_faces'")
        if values.shape != (g.ncells,) or np.any(values < 0):
            raise DataError("Cell masses must be a nonnegative vector")
        return result(values, "g")
    if name == "enclosed_mass":
        if "inner_mass" not in p or not np.isfinite(p["inner_mass"]) or p["inner_mass"] < 0:
            raise DataError("Supply a finite nonnegative inner_mass in grams")
        masses = derive(s, "cell_mass", source=p.get("source"),
                        assume_cell_average=p.get("assume_cell_average", False)).read()
        return result(p["inner_mass"] + np.r_[0., np.cumsum(masses)], "g", "face", "point")
    if name == "species_mass":
        c = s.compositions[p.get("composition", "composition")]
        x = c.mass_fractions()
        if x.dimensions != ("cell", "species"):
            raise DataError("Species masses require cell compositions")
        mass = derive(s, "cell_mass", source=p.get("source"),
                      assume_cell_average=p.get("assume_cell_average", False)).read()
        values = mass[:, None] * x.read()
        return Field(name, values, "g", location="cell", dimensions=("cell", "species"),
                     sampling="integral", valid=x.read_validity(), registry=g.registry, source=source,
                     metadata={"species": [sp.to_dict() for sp in c.species]})
    raise KeyError(name)

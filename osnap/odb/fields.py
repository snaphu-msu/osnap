"""Array fields and coordinate axes; h5py-backed values remain selective."""
from __future__ import annotations

from copy import deepcopy
import json
import numpy as np

from .errors import ClosedDatasetError, DataError
from .units import DEFAULT_UNITS


def metadata_copy(value):
    value = deepcopy({} if value is None else value)
    try:
        json.dumps(value, allow_nan=False)
    except (ValueError, TypeError) as error:
        raise DataError("Metadata must be finite, JSON-compatible values") from error
    return value


class DiskArray:
    """Private array proxy. Metadata inspection never reads the dataset payload."""

    def __init__(self, dataset, owner):
        self.dataset, self.owner = dataset, owner
        self.shape, self.dtype = dataset.shape, dataset.dtype

    def read(self, selection=None):
        if not self.owner.id.valid:
            raise ClosedDatasetError("The OSNAP HDF5 file has been closed")
        return np.asarray(self.dataset[() if selection is None else selection])


def array_read(array, selection=None):
    if isinstance(array, DiskArray):
        return array.read(selection)
    return np.array(array if selection is None else array[selection], copy=True)


class Field:
    """A numeric array with units, explicit placement, axes and source metadata.

    Values are normalized to CGS on construction, except logarithmic labels.
    ``read`` returns a detached NumPy array. It is safe for callers to edit it.
    """

    def __init__(self, name, values, unit, *, dimensions=(), location="global",
                 sampling="point", description=None, axes=None, valid=None,
                 source=None, metadata=None, registry=None, copy=True):
        self.name = str(name)
        if not self.name or "/" in self.name:
            raise DataError("Field names must be nonempty and cannot contain '/' ")
        self.registry = registry or DEFAULT_UNITS
        self.location, self.sampling = location, sampling
        if location not in ("cell", "face", "sample", "global"):
            raise DataError(f"Invalid field location {location!r}")
        if sampling not in ("point", "average", "integral"):
            raise DataError(f"Invalid sampling interpretation {sampling!r}")
        self.dimensions = tuple(dimensions)
        self.description = description or self.name.replace("_", " ")
        self.axes = dict(axes or {d: d for d in self.dimensions})
        if set(self.axes) != set(self.dimensions):
            raise DataError("Every field dimension needs exactly one axis reference")
        self.source, self.metadata = metadata_copy(source), metadata_copy(metadata)
        self.unit = self.registry.cgs_unit(unit)
        factor = self.registry.factor(unit, self.unit)
        if isinstance(values, DiskArray):
            if factor != 1:
                raise DataError("Disk-backed OSNAP fields must already be in CGS")
            self._values = values
        else:
            self._values = np.array(values, copy=True) if copy else np.asarray(values)
            if factor != 1:
                self._values = self._values.astype(np.float64)
                self._values *= factor
            self._values.flags.writeable = False
        self.shape, self.dtype = self._values.shape, self._values.dtype
        if self.dtype.kind not in "biuf":
            raise DataError("Fields require real numeric values, not strings/objects/complex numbers")
        if len(self.dimensions) != len(self.shape) or len(set(self.dimensions)) != len(self.dimensions):
            raise DataError("Dimensions must uniquely name each array axis")
        if location in ("cell", "face") and (not self.dimensions or self.dimensions[0] != location):
            raise DataError(f"A {location} field must begin with the {location!r} dimension")
        if location == "global" and self.shape:
            raise DataError("Global fields are scalar; use location='sample' for arrays")
        if valid is None or isinstance(valid, DiskArray):
            self._valid = valid
        else:
            self._valid = np.array(valid, dtype=bool, copy=True)
            self._valid.flags.writeable = False
        if self._valid is not None and self._valid.shape != self.shape:
            raise DataError("Validity mask and field shapes differ")
        if isinstance(self._valid, DiskArray) and self._valid.dtype.kind != "b":
            raise DataError("On-disk validity masks must be boolean")
        if unit != self.unit:
            self.source.setdefault("transformations", []).append(
                {"operation": "unit_conversion", "source_unit": unit,
                 "target_unit": self.unit, "factor": factor})
        if self.registry.parse(unit).kind == "logarithmic" and not self.metadata.get("magnitude_system"):
            raise DataError("Magnitude fields require metadata['magnitude_system']")

    @property
    def key(self):
        return self.location, self.name

    def read(self, selection=None, *, unit=None):
        values = array_read(self._values, selection)
        if unit is not None:
            factor = self.registry.factor(self.unit, unit)
            if factor != 1:
                values = values.astype(np.float64)
                values *= factor
        return values

    def read_validity(self, selection=None):
        if self._valid is not None:
            return array_read(self._valid, selection)
        # A zero-stride view computes the selected shape without allocating the full field.
        dummy = np.broadcast_to(np.array(True), self.shape)
        return np.array(dummy if selection is None else dummy[selection], copy=True)

    def require_valid(self):
        values = self.read()
        if not np.all(self.read_validity()) or not np.all(np.isfinite(values)):
            raise DataError(f"Field {self.name!r} has missing or non-finite required values")
        return values.astype(np.float64)

    def replaced(self, values, **changes):
        options = dict(unit=self.unit, dimensions=self.dimensions, location=self.location,
                       sampling=self.sampling, description=self.description, axes=self.axes,
                       source=self.source, metadata=self.metadata, registry=self.registry)
        options.update(changes)
        name = options.pop("name", self.name)
        return Field(name, values, **options)

    def subset(self, selection):
        return self.replaced(self.read(selection), valid=self.read_validity(selection))


class Axis:
    """One-dimensional numeric coordinate or categorical label array."""

    def __init__(self, name, values, unit="1", *, description=None, bounds=None,
                 metadata=None, registry=None):
        self.name, self.description = name, description or name
        self.metadata = metadata_copy(metadata)
        self.registry = registry or DEFAULT_UNITS
        if isinstance(values, DiskArray):
            self._values = values
            self.unit = unit
            self.registry.parse(unit)
        else:
            values = np.asarray(values)
            if values.dtype.kind in "US":
                if unit != "1":
                    raise DataError("Categorical axes must be dimensionless")
                self._values, self.unit = np.array(values, copy=True), unit
            elif values.dtype.kind in "iuf":
                field = Field(name, values, unit, dimensions=(name,), location="sample", registry=self.registry)
                self._values, self.unit = field._values, field.unit
                if field.source:
                    self.metadata.setdefault("source", field.source)
            else:
                raise DataError("Axes require numeric values or string labels")
        self.shape = self._values.shape
        if len(self.shape) != 1:
            raise DataError("Axes must be one-dimensional")
        self.bounds = None
        if bounds is not None:
            self.bounds = bounds if isinstance(bounds, Field) else Field(
                name + "_bounds", bounds, unit, dimensions=(name + "_edge",),
                location="sample", registry=self.registry)
            if self.bounds.shape != (self.shape[0] + 1,):
                raise DataError("Axis bounds need one more value than coordinates")
            self.registry.factor(self.bounds.unit, self.unit)

    def read(self, selection=None, *, unit=None):
        values = array_read(self._values, selection)
        if values.dtype.kind == "O":
            values = values.astype(str)
        if unit is not None:
            factor = self.registry.factor(self.unit, unit)
            if factor != 1:
                values = values.astype(np.float64) * factor
        return values


def field_map(fields):
    result = {}
    for field in (fields.values() if isinstance(fields, dict) else fields or []):
        if not isinstance(field, Field) or field.key in result:
            raise DataError("Fields must be Field objects with unique (location, name) keys")
        result[field.key] = field
    return result


def lookup(fields, name, location=None):
    matches = [f for (loc, n), f in fields.items() if n == name and (location is None or loc == location)]
    if not matches:
        raise KeyError(f"Missing field {name!r} at {location or 'any location'}")
    if len(matches) > 1:
        raise DataError(f"Ambiguous field {name!r}; specify location")
    return matches[0]

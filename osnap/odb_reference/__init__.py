"""OSNAP database (ODB): scientific objects, operations, and optional HDF5 persistence.

Runtime dependencies are NumPy and h5py. Importing ODB loads no legacy modules.
"""
from .errors import DataError, UnitError, FormatError, ClosedDatasetError
from .units import Unit, UnitRegistry, DEFAULT_UNITS
from .fields import Axis, Field
from .composition import Composition, Species
from .model import Dataset, Series, Snapshot, RadialGrid, Trajectory, TracerSet, Report
from .derived import Dependency, register_derived
from .adapters import Adapter

__version__ = "1.0.0"
__all__ = ["Dataset", "Series", "Snapshot", "RadialGrid", "Trajectory", "TracerSet",
           "Field", "Axis", "Composition", "Species", "Unit", "UnitRegistry", "DEFAULT_UNITS",
           "Report", "Dependency", "register_derived", "DataError", "UnitError", "FormatError",
           "ClosedDatasetError", "Adapter"]

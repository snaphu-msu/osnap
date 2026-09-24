import sys
import subprocess
import numpy as np
import pytest

from osnap.odb_reference import (Axis, Composition, DataError, Dataset, Dependency, Field, RadialGrid,
                       Series, Snapshot, Species, Trajectory, TracerSet, UnitError, UnitRegistry,
                       register_derived)


@pytest.mark.parametrize("source,target,factor", [
    ("km/s","cm/s",1e5), ("kg/m^3","g/cm^3",1e-3),
    ("erg/(g*K)","cm^2/(s^2*K)",1), ("GK","K",1e9), ("day","s",86400),
    ("MeV","erg",1.602176634e-6), ("k_B/baryon","erg/(K*baryon)",1.380649e-16),
])
def test_unit_conversions(source,target,factor):
    assert UnitRegistry().factor(source,target) == pytest.approx(factor)


@pytest.mark.parametrize("unit", ["", "cm + s", "__import__('os')", "cm**.5", "cm**100", "unknown", "cm s", "(cm", "cm//s"])
def test_unsupported_unit_syntax(unit):
    with pytest.raises(UnitError):
        UnitRegistry().parse(unit)


def test_dimensional_and_logarithmic_guards():
    units = UnitRegistry()
    for source,target in [("cm","s"),("k_B/baryon","erg/g/K"),("mag","1")]:
        with pytest.raises(UnitError):
            units.factor(source,target)
    with pytest.raises(DataError,match="magnitude_system"):
        Field("magnitude",1.,"mag")
    field = Field("magnitude",1.,"mag",metadata={"magnitude_system":"AB"})
    assert field.read() == 1


def test_field_owns_arrays_masks_and_source_conversion():
    values = np.array([1.,2.],dtype=np.float32)
    f = Field("speed",values,"km/s",dimensions=("cell",),location="cell",valid=[True,False])
    values[:] = 0
    np.testing.assert_array_equal(f.read(),[1e5,2e5])
    detached = f.read(); detached[:] = 0
    assert f.read()[0] == 1e5
    np.testing.assert_array_equal(f.read_validity(slice(1,2)),[False])
    assert f.source["transformations"][0]["factor"] == 1e5
    with pytest.raises(DataError,match="missing"):
        f.require_valid()


def test_field_shapes_and_grid_locations():
    with pytest.raises(DataError):
        Field("a",[1,2],"1")
    with pytest.raises(DataError):
        Field("a",[1,2],"1",dimensions=("cell",),location="face")
    with pytest.raises(DataError):
        RadialGrid([0,2,1])
    with pytest.raises(DataError):
        RadialGrid([0,1,2],radius_cells=[.5,3])
    with pytest.raises(DataError):
        Snapshot([Field("density",[1,2],"g/cm^3",dimensions=("cell",),location="cell")],grid=RadialGrid([0,1]))
    grid = RadialGrid([1,2,4],mass_faces=[10,11,15])
    np.testing.assert_array_equal(grid.radius_cells.read(),[1.5,3])
    assert grid.metadata["cell_sampling"] == "radial_midpoint"


def test_scalar_unit_conversion_preserves_array_interface():
    units = UnitRegistry()
    units.register("code_length", 1e8, (0, 1, 0, 0, 0))
    source = np.array(2.0, dtype=np.float32)
    field = Field("length", source, "code_length", registry=units)
    for unit, expected in (("cm", 2e8), ("code_length", 2.0)):
        values = field.read(unit=unit)
        assert isinstance(values, np.ndarray)
        assert values.shape == () and values.dtype == np.float64
        assert values.item() == expected
        values[...] = 0
    assert field.read().item() == 2e8
    assert source.item() == 2.0


def test_ambiguity_selection_and_reductions(shell_model):
    s = shell_model.with_field(Field("radial_velocity",[1,2,3],"cm/s",dimensions=("cell",),location="cell"))
    with pytest.raises(DataError,match="Ambiguous"):
        s.field("radial_velocity")
    selected = s.select_cells(1,3)
    assert selected.grid.radius_faces.read()[0] == 3
    assert selected.grid.mass_faces.read()[0] > 10
    assert selected.field("radial_velocity",location="face").shape == (3,)
    assert selected.provenance[-1]["excluded_mass"]["value"] > 0
    mass = s.derive("cell_mass",source="density")
    average = s.reduce("temperature",operation="average",weights=mass)
    assert average.read() == pytest.approx(np.average([3e7,5e7,7e7],weights=mass.read()))
    with pytest.raises(DataError):
        s.reduce("temperature",operation="average")


def test_composition_preserves_values_and_unknown_groups():
    raw = Field("X",[[.4,.4]],"1",dimensions=("cell","species"),location="cell")
    composition = Composition(raw,[Species.parse("h1"),Species.parse("he4")],complete=True)
    assert composition.abundances.read().sum() == .8
    assert composition.normalize().mass_fractions().read().sum() == 1
    np.testing.assert_allclose(composition.electron_fraction().read(),[.6])
    c = Composition(raw,[Species.parse("h1"),Species.parse("fe",groups=("fe",))])
    assert c.species[1].A is None and c.species[1].kind == "group"
    with pytest.raises(DataError):
        c.electron_fraction()
    number = Composition(raw,[Species.parse("h1"),Species.parse("he4")],basis="number_per_baryon")
    np.testing.assert_allclose(number.mass_fractions().read(),[[.4,1.6]])


def test_derived_extensions_cycle_and_placement(shell_model):
    register_derived("test_twice_density",lambda s,d: 2*d["density"],
                     dependencies=[Dependency("density","g/cm^3","cell")],unit="g/cm^3",
                     location="cell",dimensions=("cell",),description="twice density")
    np.testing.assert_allclose(shell_model.derive("test_twice_density").read(),[6,2,4])
    register_derived("test_cycle_a",lambda s,d: d["test_cycle_b"],
                     dependencies=[Dependency("test_cycle_b","1","cell",True)],unit="1",location="cell",dimensions=("cell",),description="cycle")
    register_derived("test_cycle_b",lambda s,d: d["test_cycle_a"],
                     dependencies=[Dependency("test_cycle_a","1","cell",True)],unit="1",location="cell",dimensions=("cell",),description="cycle")
    with pytest.raises(DataError,match="cycle"):
        shell_model.derive("test_cycle_a")
    point_rho = shell_model.field("density").replaced([3,1,2],sampling="point")
    s = shell_model.with_field(point_rho)
    with pytest.raises(DataError,match="shell average"):
        s.derive("cell_mass",source="density")
    s.derive("cell_mass",source="density",assume_cell_average=True)


def test_tracer_histories_and_yield_mass_requirement():
    t = Trajectory(42,[0,1,3],[Field("temperature",[1,3,7],"K",dimensions=("time",),location="sample")])
    np.testing.assert_allclose(t.interpolate([.5,2],fields=["temperature"]).field("temperature").read(),[2,5])
    with pytest.raises(DataError,match="represented mass"):
        TracerSet([t]).species_yields()
    with pytest.raises(DataError):
        t.interpolate([-1],fields=["temperature"])
    with pytest.raises(DataError):
        TracerSet([t,t])


def test_import_is_independent_of_legacy_dependencies():
    code = '''
import sys
class Deny:
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'pandas','xarray','yt','scipy','astropy','progs','flashbang','nucleosynth','yaml'}:
            raise RuntimeError('unexpected dependency '+fullname)
sys.meta_path.insert(0,Deny())
from osnap.odb import Dataset, Field, Adapter
assert 'osnap.config' not in sys.modules
'''
    subprocess.run([sys.executable,"-c",code],check=True)

import numpy as np
import pytest
from osnap.odb import Composition, DataError, Field, RadialGrid, Series, Snapshot


@pytest.mark.parametrize("coordinate",["radius","mass"])
@pytest.mark.parametrize("mesh",["identity","refine","coarsen","irregular"])
def test_conservation_and_positivity(shell_model,coordinate,mesh):
    old = shell_model.grid.radius_faces.read() if coordinate == "radius" else shell_model.grid.mass_faces.read()
    targets = {"identity":old, "refine":np.sort(np.r_[old,(old[:-1]+old[1:])/2]),
               "coarsen":old[[0,-1]],"irregular":np.linspace(old[0],old[-1],17)}
    result = shell_model.remap(targets[mesh],coordinate=coordinate,mass_source="mass_faces",
                              field_policies={"temperature":"mass_weighted","radial_velocity":"point"})
    before = shell_model.derive("cell_mass",source="mass_faces").read()
    after = result.derive("cell_mass",source="density").read()
    np.testing.assert_allclose(after.sum(),before.sum(),rtol=1e-12,atol=0)
    np.testing.assert_allclose(result.derive("species_mass",source="density").read().sum(axis=0),
                               shell_model.derive("species_mass",source="mass_faces").read().sum(axis=0),rtol=1e-12,atol=0)
    assert np.all(result.compositions["composition"].mass_fractions().read() >= 0)
    np.testing.assert_allclose(result.compositions["composition"].mass_fractions().read().sum(axis=1),1)
    assert result.grid.radius_faces.read()[0] == shell_model.grid.radius_faces.read()[0]
    assert result.grid.mass_faces.read()[0] == 10
    assert result.grid.ncells == len(targets[mesh])-1


def test_remap_does_not_repair_incomplete_abundances(shell_model):
    c = shell_model.compositions["composition"]
    incomplete = Composition(c.abundances.replaced(c.abundances.read()*.4),c.species,complete=False)
    s = Snapshot(shell_model.fields,grid=shell_model.grid,compositions={"composition":incomplete})
    out = s.remap([2,8],coordinate="radius",mass_source="density",field_policies={})
    assert out.compositions["composition"].mass_fractions().read().sum() == pytest.approx(.4)
    assert not out.compositions["composition"].complete


def test_zero_mass_and_invalid_samples(shell_model):
    empty = shell_model.with_field(shell_model.field("density").replaced([0.,0.,0.]))
    out = empty.remap([2,8],coordinate="radius",mass_source="density",field_policies={})
    assert out.field("density").read()[0] == 0
    assert not out.compositions["composition"].abundances.read_validity().any()
    invalid = shell_model.with_field(shell_model.field("density").replaced([1,2,3],valid=[True,False,True]))
    with pytest.raises(DataError):
        invalid.remap([2,8],coordinate="radius",mass_source="density",field_policies={})
    with pytest.raises(DataError,match="source domain"):
        shell_model.remap([3,8],coordinate="radius",mass_source="density",field_policies={})


def test_explicit_centering_boundaries(shell_model):
    centers = shell_model.grid.radius_cells.read()
    s = shell_model.with_field(Field("linear",2*centers+1,"K",location="cell",dimensions=("cell",)))
    with pytest.raises(DataError,match="boundary"):
        s.interpolate_field("linear",target_location="face")
    f = s.interpolate_field("linear",target_location="face",boundary="linear")
    np.testing.assert_allclose(f.read(),2*shell_model.grid.radius_faces.read()+1)
    back = s.with_field(f).interpolate_field("linear",location="face",target_location="cell")
    np.testing.assert_allclose(back.read(),2*centers+1)
    explicit = s.interpolate_field("linear",target_location="face",boundary=(5.,17.))
    np.testing.assert_allclose(explicit.read(),f.read())


def test_time_interpolation_blends_species_masses_not_fractions(shell_model):
    a = shell_model
    c = a.compositions["composition"]
    b = Snapshot([a.field("density").replaced(a.field("density").read()*3)],grid=a.grid,
                 compositions={"composition":Composition(c.abundances.replaced(1-c.abundances.read()),c.species,complete=True)},time=2.)
    series = Series([a,b])
    target = RadialGrid([2,4,8])
    result = series.interpolate(1.,target_grid=target,coordinate="radius",field_policies={},mass_source="density")
    expected = (a.derive("species_mass",source="density").read().sum(axis=0)+b.derive("species_mass",source="density").read().sum(axis=0))/2
    np.testing.assert_allclose(result.derive("species_mass",source="density").read().sum(axis=0),expected,rtol=1e-12,atol=0)
    with pytest.raises(DataError,match="references"):
        series.interpolate(1,target_grid=target,coordinate="radius",field_policies={},mass_source="density",time_reference="explosion")
    shifted = series.shifted_time(-1,time_reference="bounce")
    assert shifted.snapshot(0).time == -1

import json
import h5py
import numpy as np
import pytest
from osnap.odb_reference import (Axis, ClosedDatasetError, Dataset, Field, FormatError, Series, Snapshot,
                       Trajectory, TracerSet, UnitRegistry)


def test_roundtrip_selective_access_and_closed_file(tmp_path,shell_model,monkeypatch):
    path = tmp_path/"model.osnap.h5"
    original = Dataset(series={"hydrodynamics":Series([shell_model])},metadata={"unknown_extension":{"note":"retain me"}},provenance=[{"user":"test"}])
    original.write(path)
    selections = []
    read = h5py.Dataset.__getitem__
    def tracked(dataset,selection):
        if dataset.name.endswith("/values"):
            selections.append((dataset.name,selection))
        return read(dataset,selection)
    monkeypatch.setattr(h5py.Dataset,"__getitem__",tracked)
    with Dataset.open(path) as loaded:
        assert selections == []
        snapshot = loaded.series["hydrodynamics"].snapshot(0)
        assert selections == []
        field = snapshot.field("density")
        values = field.read(slice(1,2))
        np.testing.assert_array_equal(values,[1.])
        assert len(selections) == 1 and selections[0][1] == slice(1,2)
        assert loaded.metadata == original.metadata
        assert loaded.provenance == original.provenance
        c = snapshot.compositions["composition"]
        assert c.species == shell_model.compositions["composition"].species
        loaded.write(tmp_path/"copy.h5")
    with pytest.raises(ClosedDatasetError):
        field.read()
    with pytest.raises(ClosedDatasetError):
        loaded.series["hydrodynamics"].snapshot(0)
    np.testing.assert_array_equal(values,[1.])


def test_ragged_tracers_products_masks_and_custom_units(tmp_path):
    reg = UnitRegistry(); reg.register("code_length",7,(0,1,0,0,0))
    tracer1 = Trajectory(300,[0.,1.],[Field("radius",[1,2],"code_length",dimensions=("time",),location="sample",valid=[True,False],registry=reg)],represented_mass=Field("represented_mass",2,"g"))
    tracer2 = Trajectory("other/id",[2.,3.,7.],[Field("radius",[3,4,5],"cm",dimensions=("time",),location="sample")])
    bands = Axis("band",["U","B","V"])
    mags = Field("magnitude",[1.,2.,3.],"mag",dimensions=("band",),location="sample",metadata={"magnitude_system":"Vega"})
    data = Dataset(tracers={"tracers":TracerSet([tracer1,tracer2])},series={"photometry":Series([Snapshot([mags],axes={"band":bands},time=3.)])})
    path = tmp_path/"ragged.h5"; data.write(path)
    with Dataset.open(path) as loaded:
        np.testing.assert_allclose(loaded.tracers["tracers"].trajectory(300).field("radius").read(unit="code_length"),[1,2])
        np.testing.assert_array_equal(loaded.tracers["tracers"].trajectory(300).field("radius").read_validity(),[True,False])
        assert loaded.tracers["tracers"].trajectory("other/id").time.shape == (3,)
        np.testing.assert_array_equal(loaded.series["photometry"].snapshot(0).axes["band"].read(),["U","B","V"])


def test_no_overwrite_and_unknown_schema(tmp_path,shell_model):
    path=tmp_path/"data.h5"
    dataset=Dataset(series={"model":Series([shell_model])})
    dataset.write(path)
    with pytest.raises(FileExistsError):
        dataset.write(path)
    with h5py.File(path,"r+") as f:
        f.attrs["schema_version"]="99.0"
    with pytest.raises(FormatError):
        Dataset.open(path)


def test_failed_write_does_not_replace_file(tmp_path,shell_model,monkeypatch):
    path=tmp_path/"data.h5"
    dataset=Dataset(series={"model":Series([shell_model])})
    dataset.write(path)
    before=path.read_bytes()
    from osnap.odb_reference import storage
    def fail(*a,**kw):
        raise RuntimeError("injected write failure")
    monkeypatch.setattr(storage,"_snapshot_write",fail)
    with pytest.raises(RuntimeError):
        dataset.write(path,overwrite=True)
    assert path.read_bytes()==before
    assert not list(tmp_path.glob("*.tmp"))

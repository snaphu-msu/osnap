from pathlib import Path
import numpy as np
import pytest

from osnap.odb_reference import Composition, Field, RadialGrid, Snapshot, Species

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def shell_model():
    radius = np.array([2., 3., 5., 8.])
    volume = 4*np.pi/3*np.diff(radius**3)
    rho = np.array([3., 1., 2.])
    mass = rho*volume
    grid = RadialGrid(radius, mass_faces=10.+np.r_[0.,np.cumsum(mass)], frame="lagrangian")
    x = np.array([[1.,0.], [.5,.5], [0.,1.]])
    c = Composition(Field("abundances",x,"1",dimensions=("cell","species"), location="cell",sampling="average"),
                    [Species.parse("h1"),Species.parse("he4")],complete=True)
    return Snapshot([
        Field("density",rho,"g/cm^3",dimensions=("cell",),location="cell",sampling="average"),
        Field("temperature",np.array([3.,5.,7.])*1e7,"K",dimensions=("cell",),location="cell",sampling="average"),
        Field("radial_velocity",radius*2,"cm/s",dimensions=("face",),location="face"),
        Field("electron_fraction",np.sum(x*np.array([1.,.5]),axis=1),"1",dimensions=("cell",),location="cell",sampling="average"),
    ],grid=grid,compositions={"composition":c},time=0.,metadata={"user_annotation":{"keep":True}})

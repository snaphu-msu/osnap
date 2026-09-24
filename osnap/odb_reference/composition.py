"""Species identity and explicit abundance conversions (no automatic repair)."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import re
import numpy as np

from .errors import DataError
from .fields import Field, metadata_copy

ELEMENTS = ("n H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
            "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd "
            "Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac "
            "Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl Mc Lv Ts Og").split()


@dataclass(frozen=True)
class Species:
    id: str
    source_label: str
    kind: str = "isotope"
    Z: int | None = None
    A: int | None = None
    description: str = ""

    def __post_init__(self):
        if not self.id or self.kind not in ("isotope", "element", "group"):
            raise DataError("Invalid species identity/kind")
        if self.Z is not None and (int(self.Z) != self.Z or not 0 <= self.Z <= 118):
            raise DataError("Invalid atomic number")
        if self.kind == "isotope" and (self.Z is None or self.A is None or int(self.A) != self.A
                                      or self.A < max(1, self.Z)):
            raise DataError("Isotopes require physical integer Z and A")
        if self.kind == "element" and (self.Z is None or self.Z < 1 or self.A is not None):
            raise DataError("Elements require Z and no mass number")
        if self.kind == "group" and not self.description:
            raise DataError("Aggregate species require a description")

    @classmethod
    def isotope(cls, Z, A, source_label=None):
        Z, A = int(Z), int(A)
        if not 0 <= Z < len(ELEMENTS):
            raise DataError("Invalid atomic number")
        label = "n" if Z == 0 and A == 1 else f"{ELEMENTS[Z].lower()}{A}"
        return cls(label, source_label or label, Z=Z, A=A)

    @classmethod
    def parse(cls, label, *, groups=()):
        clean = label.strip().strip("'").lower()
        if clean in groups:
            return cls(f"group:{clean}", label, "group", description=f"Source-defined {label} aggregate")
        aliases = {"neut": (0, 1), "neutrons": (0, 1), "nt1": (0, 1), "n": (0, 1),
                   "prot": (1, 1), "p": (1, 1), "d": (1, 2), "t": (1, 3)}
        if clean in aliases:
            return cls.isotope(*aliases[clean], source_label=label)
        match = re.fullmatch(r"([a-z]+)(\d+)?", clean)
        if not match or match[1] not in [s.lower() for s in ELEMENTS[1:]]:
            raise DataError(f"Unrecognized species {label!r}; supply an explicit species definition")
        Z = [s.lower() for s in ELEMENTS].index(match[1])
        if match[2]:
            return cls.isotope(Z, int(match[2]), source_label=label)
        return cls(f"element:{Z}", label, "element", Z=Z)

    def to_dict(self):
        return asdict(self)


class Composition:
    def __init__(self, abundances, species, *, basis="mass_fraction", complete=False,
                 interpretation="exclusive", metadata=None):
        self.abundances, self.species = abundances, tuple(species)
        self.basis, self.complete = basis, bool(complete)
        self.interpretation, self.metadata = interpretation, metadata_copy(metadata)
        if basis not in ("mass_fraction", "number_per_baryon"):
            raise DataError("Unsupported abundance convention")
        if interpretation not in ("exclusive", "elemental_totals", "residual"):
            raise DataError("Specify exclusive, elemental_totals, or residual composition")
        if not abundances.dimensions or abundances.dimensions[-1] != "species":
            raise DataError("Composition needs a final species dimension")
        if abundances.shape[-1] != len(self.species) or not self.species:
            raise DataError("Species table and abundance axis differ")
        if len({s.id for s in self.species}) != len(self.species):
            raise DataError("Duplicate species IDs")
        abundances.registry.factor(abundances.unit, "1")
        if interpretation == "exclusive":
            element_z = {s.Z for s in self.species if s.kind == "element"}
            if any(s.kind == "isotope" and s.Z in element_z for s in self.species):
                raise DataError("Element totals and their isotopes must be separate composition tables")

    def mass_fractions(self):
        values = self.abundances.read().astype(np.float64)
        if self.basis == "number_per_baryon":
            if any(s.kind != "isotope" for s in self.species):
                raise DataError("Number-to-mass conversion requires isotope mass numbers")
            values *= np.array([s.A for s in self.species])
        return self.abundances.replaced(values, name="mass_fraction",
            valid=self.abundances.read_validity(), source={**self.abundances.source,
            "abundance_conversion": self.basis + " -> mass_fraction"})

    def electron_fraction(self):
        if not self.complete or any(s.kind != "isotope" for s in self.species):
            raise DataError("Electron fraction requires a complete isotopic composition")
        fractions = self.mass_fractions()
        values = np.sum(fractions.read() * np.array([s.Z/s.A for s in self.species]), axis=-1)
        dims = fractions.dimensions[:-1]
        return Field("composition_electron_fraction", values, "1", dimensions=dims,
                     location=fractions.location if dims else "global", sampling=fractions.sampling,
                     axes={d: fractions.axes[d] for d in dims}, valid=np.all(fractions.read_validity(), axis=-1),
                     source={"derivation": "sum(X_i Z_i/A_i)"}, registry=fractions.registry)

    def normalize(self):
        fractions = self.mass_fractions()
        values = fractions.require_valid()
        total = values.sum(axis=-1, keepdims=True)
        if np.any(values < 0) or np.any(total <= 0):
            raise DataError("Normalization requires nonnegative abundances and positive sums")
        field = fractions.replaced(values / total,
                                   source={**fractions.source, "operation": "normalize"})
        return Composition(field, self.species, complete=self.complete,
                           interpretation=self.interpretation,
                           metadata={**self.metadata, "normalization": "explicit"})

    def elemental_sums(self):
        if any(s.Z is None for s in self.species) or self.interpretation == "residual":
            raise DataError("Elemental totals require unambiguous atomic numbers and exclusive abundances")
        field = self.mass_fractions()
        zs = sorted({s.Z for s in self.species})
        values = np.stack([field.read()[..., [s.Z == z for s in self.species]].sum(axis=-1) for z in zs], axis=-1)
        valid = np.stack([field.read_validity()[..., [s.Z == z for s in self.species]].all(axis=-1) for z in zs], axis=-1)
        species = [Species.isotope(0, 1) if z == 0 else Species(f"element:{z}", ELEMENTS[z], "element", Z=z) for z in zs]
        return Composition(field.replaced(values, valid=valid), species,
                           complete=self.complete, interpretation="elemental_totals",
                           metadata={**self.metadata, "operation": "elemental_sum"})

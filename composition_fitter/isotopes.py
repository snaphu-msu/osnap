"""Lightweight isotope metadata helpers.

This intentionally stays independent from ``skynet_tools.nucleo_helpers`` so the
composition fitter does not require a SkyNet installation just to parse isotope
names into nuclear metadata.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ELEMENT_SYMBOLS = [
    "h",
    "he",
    "li",
    "be",
    "b",
    "c",
    "n",
    "o",
    "f",
    "ne",
    "na",
    "mg",
    "al",
    "si",
    "p",
    "s",
    "cl",
    "ar",
    "k",
    "ca",
    "sc",
    "ti",
    "v",
    "cr",
    "mn",
    "fe",
    "co",
    "ni",
    "cu",
    "zn",
    "ga",
    "ge",
    "as",
    "se",
    "br",
    "kr",
    "rb",
    "sr",
    "y",
    "zr",
    "nb",
    "mo",
    "tc",
    "ru",
    "rh",
    "pd",
    "ag",
    "cd",
    "in",
    "sn",
    "sb",
    "te",
    "i",
    "xe",
    "cs",
    "ba",
    "la",
    "ce",
    "pr",
    "nd",
    "pm",
    "sm",
    "eu",
    "gd",
    "tb",
    "dy",
    "ho",
    "er",
    "tm",
    "yb",
    "lu",
    "hf",
    "ta",
    "w",
    "re",
    "os",
    "ir",
    "pt",
    "au",
    "hg",
    "tl",
    "pb",
    "bi",
    "po",
    "at",
    "rn",
    "fr",
    "ra",
    "ac",
    "th",
    "pa",
    "u",
    "np",
    "pu",
    "am",
    "cm",
    "bk",
    "cf",
    "es",
    "fm",
    "md",
    "no",
    "lr",
    "rf",
    "db",
    "sg",
    "bh",
    "hs",
    "mt",
    "ds",
    "rg",
    "cn",
    "ut",
    "fl",
    "up",
    "lv",
    "us",
    "uo",
]

SYMBOL_TO_Z = {symbol: index + 1 for index, symbol in enumerate(ELEMENT_SYMBOLS)}
ISOTOPE_RE = re.compile(r"([a-z]+)(\d+)$")


@dataclass(frozen=True)
class Isotope:
    """Parsed nuclear species metadata."""

    name: str
    a: int
    z: int

    @property
    def z_over_a(self) -> float:
        return self.z / self.a


def parse_isotope(name: str) -> Isotope:
    """Parse an isotope name like ``si28`` or the special neutron label ``nt1``."""

    if name == "nt1":
        return Isotope(name=name, a=1, z=0)

    match = ISOTOPE_RE.fullmatch(name)
    if match is None:
        raise ValueError(f"Unsupported isotope label: {name!r}")

    symbol, a_text = match.groups()
    try:
        z = SYMBOL_TO_Z[symbol]
    except KeyError as exc:
        raise ValueError(f"Unknown element symbol in isotope label: {name!r}") from exc

    return Isotope(name=name, a=int(a_text), z=z)

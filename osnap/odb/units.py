"""Small multiplicative unit registry; dimensions are (M, L, T, K, baryon).

The parser accepts only registered names, 1, *, /, parentheses and signed
integer powers (** or ^). No expression is executed. Magnitudes are labels,
and deliberately cannot be converted to dimensionless values or fluxes.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

from .errors import UnitError


@dataclass(frozen=True)
class Unit:
    scale: float
    dimensions: tuple[int, ...]
    kind: str = "linear"

    def multiply(self, other, sign=1):
        if self.kind != "linear" or other.kind != "linear":
            raise UnitError("Logarithmic unit labels cannot form compound units")
        return Unit(self.scale * other.scale**sign,
                    tuple(a + sign*b for a, b in zip(self.dimensions, other.dimensions)))

    def power(self, exponent):
        if self.kind != "linear":
            raise UnitError("Logarithmic unit labels cannot be raised to powers")
        return Unit(self.scale**exponent, tuple(exponent*x for x in self.dimensions))


class UnitRegistry:
    """Versioned constants and explicit conversion factors, independent of Astropy."""

    version = "OSNAP-units-1"

    def __init__(self, definitions=None):
        self.definitions = {}
        if definitions is not None:
            for name, definition in definitions.items():
                self.register(name, definition["scale"], definition["dimensions"],
                              kind=definition.get("kind", "linear"))
            return
        for name, scale, dims in [
            ("1", 1, (0,0,0,0,0)), ("g", 1, (1,0,0,0,0)),
            ("kg", 1e3, (1,0,0,0,0)), ("M_sun", 1.988409870698051e33, (1,0,0,0,0)),
            ("cm", 1, (0,1,0,0,0)), ("m", 100, (0,1,0,0,0)),
            ("km", 1e5, (0,1,0,0,0)), ("nm", 1e-7, (0,1,0,0,0)),
            ("Angstrom", 1e-8, (0,1,0,0,0)), ("R_sun", 6.957e10, (0,1,0,0,0)),
            ("s", 1, (0,0,1,0,0)), ("ms", 1e-3, (0,0,1,0,0)),
            ("us", 1e-6, (0,0,1,0,0)), ("day", 86400, (0,0,1,0,0)),
            ("yr", 31557600, (0,0,1,0,0)), ("K", 1, (0,0,0,1,0)),
            ("GK", 1e9, (0,0,0,1,0)), ("erg", 1, (1,2,-2,0,0)),
            ("J", 1e7, (1,2,-2,0,0)), ("eV", 1.602176634e-12, (1,2,-2,0,0)),
            ("keV", 1.602176634e-9, (1,2,-2,0,0)), ("MeV", 1.602176634e-6, (1,2,-2,0,0)),
            ("Hz", 1, (0,0,-1,0,0)), ("dyn", 1, (1,1,-2,0,0)),
            ("Pa", 10, (1,-1,-2,0,0)), ("L_sun", 3.828e33, (1,2,-3,0,0)),
            ("rad", 1, (0,0,0,0,0)), ("sr", 1, (0,0,0,0,0)),
            ("baryon", 1, (0,0,0,0,1)), ("k_B", 1.380649e-16, (1,2,-2,-1,0)),
        ]:
            self.register(name, scale, dims)
        self.register("mag", 1, (0,0,0,0,0), kind="logarithmic")

    def register(self, name, scale, dimensions, *, kind="linear"):
        if name != "1" and not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
            raise UnitError(f"Invalid unit symbol: {name!r}")
        if not math.isfinite(scale) or scale <= 0 or len(dimensions) != 5:
            raise UnitError("Units require positive finite scale and five exponents")
        if any(isinstance(x, bool) or int(x) != x for x in dimensions):
            raise UnitError("Dimensional exponents must be integers")
        if kind not in ("linear", "logarithmic"):
            raise UnitError(f"Unknown unit kind {kind}")
        value = Unit(float(scale), tuple(int(x) for x in dimensions), kind)
        if name in self.definitions and self.definitions[name] != value:
            raise UnitError(f"Cannot redefine unit {name}")
        self.definitions[name] = value

    def parse(self, expression):
        if not isinstance(expression, str) or not expression.strip():
            raise UnitError("Units must be a nonempty string; use '1' for dimensionless")
        tokens = re.findall(r"\*\*|[A-Za-z_][A-Za-z_0-9]*|[+-]?\d+|[*/()^]", expression)
        if "".join(tokens) != re.sub(r"\s+", "", expression):
            raise UnitError(f"Unsupported unit syntax: {expression!r}")
        pos = 0

        def factor():
            nonlocal pos
            if pos >= len(tokens):
                raise UnitError("Incomplete unit expression")
            token = tokens[pos]
            pos += 1
            if token == "(":
                result = product()
                if pos >= len(tokens) or tokens[pos] != ")":
                    raise UnitError("Unbalanced unit parentheses")
                pos += 1
            else:
                try:
                    result = self.definitions[token]
                except KeyError:
                    raise UnitError(f"Unknown unit symbol {token!r}") from None
            if pos < len(tokens) and tokens[pos] in ("^", "**"):
                pos += 1
                if pos >= len(tokens) or not re.fullmatch(r"[+-]?\d+", tokens[pos]):
                    raise UnitError("A unit power must be an integer")
                exponent = int(tokens[pos])
                if abs(exponent) > 32:
                    raise UnitError("Unit exponent exceeds supported range [-32, 32]")
                pos += 1
                result = result.power(exponent)
            return result

        def product():
            nonlocal pos
            result = factor()
            while pos < len(tokens) and tokens[pos] in ("*", "/"):
                sign = 1 if tokens[pos] == "*" else -1
                pos += 1
                result = result.multiply(factor(), sign)
            return result

        try:
            result = product()
        except (OverflowError, ZeroDivisionError) as error:
            raise UnitError("Unit scale overflow") from error
        if pos != len(tokens) or not math.isfinite(result.scale) or result.scale <= 0:
            raise UnitError(f"Invalid unit expression {expression!r}")
        return result

    def factor(self, source, target):
        a, b = self.parse(source), self.parse(target)
        if a.dimensions != b.dimensions or a.kind != b.kind:
            raise UnitError(f"Incompatible units: {source!r} and {target!r}")
        if a.kind != "linear" and source != target:
            raise UnitError("Logarithmic conversions require a physical operation")
        return a.scale / b.scale

    def cgs_unit(self, unit):
        parsed = self.parse(unit)
        if parsed.kind != "linear":
            return unit
        parts = [name if power == 1 else f"{name}^{power}"
                 for name, power in zip(("g", "cm", "s", "K", "baryon"), parsed.dimensions) if power]
        return "*".join(parts) or "1"

    def to_dict(self):
        return {name: {"scale": u.scale, "dimensions": list(u.dimensions), "kind": u.kind}
                for name, u in self.definitions.items()}


DEFAULT_UNITS = UnitRegistry()

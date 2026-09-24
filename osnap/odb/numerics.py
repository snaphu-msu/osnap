"""Explicit interpolation and first-order conservative spherical remapping.

An overlap sweep uses O(N_source + N_target) memory, not a dense overlap matrix.
"""
from __future__ import annotations

import numpy as np

from .errors import DataError
from .fields import Field
from .composition import Composition
from .model import RadialGrid, Snapshot


def linear(x, values, target, *, boundary=None):
    x, values, target = (np.asarray(a, dtype=np.float64) for a in (x, values, target))
    if x.ndim != 1 or not len(x) or values.shape[0] != len(x) or np.any(np.diff(x) <= 0):
        raise DataError("Interpolation requires a nonempty, strictly increasing coordinate")
    if not all(np.all(np.isfinite(a)) for a in (x, values, target)):
        raise DataError("Interpolation cannot use missing/non-finite samples")
    outside = np.any(target < x[0]) or np.any(target > x[-1])
    if boundary not in (None, "constant", "linear"):
        raise DataError("Boundary rule must be None, 'constant', or 'linear'")
    if outside and boundary is None:
        raise DataError("Interpolation outside sampled coordinates requires an explicit boundary rule")
    if len(x) == 1:
        if outside and boundary != "constant":
            raise DataError("A single sample only supports constant boundary extension")
        return np.broadcast_to(values[0], target.shape + values.shape[1:]).copy()
    indices = np.clip(np.searchsorted(x, target, side="right")-1, 0, len(x)-2)
    frac = (target-x[indices])/(x[indices+1]-x[indices])
    if boundary == "constant":
        frac = np.clip(frac, 0, 1)
    frac = frac.reshape(target.shape+(1,)*(values.ndim-1))
    return values[indices]*(1-frac)+values[indices+1]*frac


def center_field(snapshot, f, target_location, boundary):
    if snapshot.grid is None or f.location not in ("cell", "face") or target_location not in ("cell", "face"):
        raise DataError("Centering conversion requires a grid and cell/face locations")
    if f.sampling != "point":
        raise DataError("Only point fields can be linearly relocated")
    grid = snapshot.grid
    x = grid.radius_cells.read() if f.location == "cell" else grid.radius_faces.read()
    target = grid.radius_cells.read() if target_location == "cell" else grid.radius_faces.read()
    values = f.require_valid()
    rule = boundary
    if isinstance(boundary, (tuple, list)):
        if f.location != "cell" or len(boundary) != 2:
            raise DataError("Endpoint values apply to cell-to-face conversion")
        if x[0] <= target[0] or x[-1] >= target[-1]:
            raise DataError("Explicit endpoint values require strictly interior cell samples")
        x = np.r_[target[0], x, target[-1]]
        values = np.concatenate((np.broadcast_to(boundary[0], (1,)+values.shape[1:]), values,
                                 np.broadcast_to(boundary[1], (1,)+values.shape[1:])))
        rule = None
    dims = (target_location,)+f.dimensions[1:]
    axes = {d: f.axes[d] for d in f.dimensions[1:]}
    axes[target_location] = target_location
    return f.replaced(linear(x, values, target, boundary=rule), location=target_location,
                      dimensions=dims, axes=axes, source={**f.source,
                      "centering_conversion": {"from": f.location, "to": target_location, "boundary": boundary}})


def _overlaps(source, target):
    """Return target indices, source indices and overlapping coordinate lengths."""
    i = j = 0
    si, tj, lengths = [], [], []
    while i < len(source)-1 and j < len(target)-1:
        left, right = max(source[i], target[j]), min(source[i+1], target[j+1])
        if right > left:
            si.append(i); tj.append(j); lengths.append(right-left)
        if source[i+1] <= target[j+1]:
            i += 1
        else:
            j += 1
    return np.array(tj, dtype=int), np.array(si, dtype=int), np.array(lengths)


def remap(snapshot, target_edges, *, coordinate, mass_source, field_policies,
          assume_cell_average=False, inner_mass=None):
    g = snapshot.grid
    if g is None:
        raise DataError("Remapping requires a grid")
    g.validate()
    edges = np.asarray(target_edges, dtype=np.float64).copy()
    if edges.ndim != 1 or len(edges) < 2 or not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
        raise DataError("Target boundaries must be finite and strictly increasing")
    masses = snapshot.derive("cell_mass", source=mass_source,
                            assume_cell_average=assume_cell_average).read()
    volumes = g.volumes()
    if coordinate == "radius":
        original = g.radius_faces.read()
    elif coordinate == "mass":
        if g.mass_faces is None or mass_source != "mass_faces":
            raise DataError("Mass-coordinate remapping requires mass_source='mass_faces'")
        original = g.mass_faces.read()
        if np.any(np.diff(original) <= 0):
            raise DataError("Mass-coordinate remapping requires positive source shell masses")
    else:
        raise DataError("Remap coordinate must be 'radius' or 'mass'")
    if not np.allclose(edges[[0,-1]], original[[0,-1]], rtol=1e-13, atol=0):
        raise DataError("Target must cover exactly the source domain; select a region separately")
    edges[[0,-1]] = original[[0,-1]]
    # Offset cumulative volumes avoids subtracting large absolute r**3 values.
    source_q = np.r_[0., np.cumsum(volumes)] if coordinate == "radius" else original
    if coordinate == "radius":
        target_v = 4*np.pi/3*np.diff(edges)*(edges[1:]**2+edges[1:]*edges[:-1]+edges[:-1]**2)
        target_q = np.r_[0., np.cumsum(target_v)]
        target_q[-1] = source_q[-1]
    else:
        target_q = edges
    ti, si, length = _overlaps(source_q, target_q)
    fraction = length/np.diff(source_q)[si]
    count = len(edges)-1

    def transfer(values):
        values = np.asarray(values, dtype=np.float64)
        result = np.zeros((count,)+values.shape[1:])
        weight = fraction.reshape((-1,)+(1,)*(values.ndim-1))
        np.add.at(result, ti, values[si]*weight)
        return result

    target_mass, carried_volume = transfer(masses), transfer(volumes)
    if coordinate == "mass":
        target_v = carried_volume
        # The cube root reconstruction retains the inner boundary and domain.
        r0 = g.radius_faces.read()[0]
        radii = np.cbrt(r0**3+3/(4*np.pi)*np.r_[0., np.cumsum(target_v)])
        radii[[0,-1]] = g.radius_faces.read()[[0,-1]]
        mfaces = edges
    else:
        radii = edges
        if inner_mass is None and g.mass_faces is not None:
            inner_mass = float(g.mass_faces.read(0))
        if inner_mass is not None and (not np.isfinite(inner_mass) or inner_mass < 0):
            raise DataError("inner_mass must be nonnegative finite grams")
        mfaces = None if inner_mass is None else inner_mass + np.r_[0., np.cumsum(target_mass)]
    new_grid = RadialGrid(radii, mass_faces=mfaces, frame=g.frame, registry=g.registry)
    density = Field("density", target_mass/new_grid.volumes(), "g/cm^3", dimensions=("cell",),
                    location="cell", sampling="average", registry=g.registry)
    fields = [density]
    selected_keys = {density.key}
    for key, policy in field_policies.items():
        f = snapshot.field(key) if isinstance(key, str) else snapshot.field(key[1], location=key[0])
        if f.key in selected_keys:
            raise DataError("Density is reconstructed from mass, not remapped by a field policy")
        selected_keys.add(f.key)
        values = f.require_valid()
        if f.location == "global":
            if policy != "copy":
                raise DataError("Global fields require policy='copy'")
            fields.append(f)
            continue
        if policy == "point":
            if f.sampling != "point" or f.location not in ("cell", "face"):
                raise DataError("Point interpolation requires a point-sampled cell/face field")
            if coordinate == "radius":
                sx = g.radius_faces.read() if f.location == "face" else g.radius_cells.read()
                tx = new_grid.radius_faces.read() if f.location == "face" else new_grid.radius_cells.read()
            else:
                sx = original if f.location == "face" else (original[:-1]+original[1:])/2
                tx = edges if f.location == "face" else (edges[:-1]+edges[1:])/2
            out = linear(sx, values, tx)
            valid = None
        else:
            if f.location != "cell":
                raise DataError("Only cell fields support conservative or weighted transfer")
            if policy == "integral" and f.sampling == "integral":
                out, valid = transfer(values), None
            elif policy in ("mass_weighted", "volume_weighted") and f.sampling == "average":
                weights = masses if policy == "mass_weighted" else volumes
                denom = transfer(weights).reshape((count,)+(1,)*(values.ndim-1))
                numerator = transfer(values*weights.reshape((-1,)+(1,)*(values.ndim-1)))
                out = np.divide(numerator, denom, out=np.full_like(numerator, np.nan), where=denom>0)
                valid = np.broadcast_to(denom > 0, out.shape)
            else:
                raise DataError(f"Unsupported policy/sampling combination: {policy}/{f.sampling}")
        fields.append(f.replaced(out, valid=valid, source={**f.source, "remap_policy": policy}))
    compositions = {}
    for name, c in snapshot.compositions.items():
        x = c.mass_fractions()
        if x.dimensions != ("cell", "species"):
            raise DataError("Remapping requires cell-based compositions")
        values = x.require_valid()
        if np.any(values < 0):
            raise DataError("Conservative remapping requires nonnegative abundances")
        total = transfer(values*masses[:, None])
        out = np.divide(total, target_mass[:, None], out=np.full_like(total, np.nan), where=target_mass[:, None]>0)
        if c.basis == "number_per_baryon":
            out /= np.array([s.A for s in c.species])
        compositions[name] = Composition(c.abundances.replaced(out, valid=np.broadcast_to(target_mass[:, None]>0, out.shape)),
            c.species, basis=c.basis, complete=c.complete, interpretation=c.interpretation, metadata=c.metadata)
    omitted = [f"{loc}:{name}" for loc, name in snapshot.fields if (loc, name) not in selected_keys]
    operation = {"operation": "conservative_remap", "coordinate": coordinate, "mass_source": mass_source,
                 "assume_cell_average": assume_cell_average, "omitted_fields": omitted,
                 "policies": {str(k): v for k, v in field_policies.items()},
                 "source_mass_g": float(masses.sum()), "target_mass_g": float(target_mass.sum())}
    return Snapshot(fields, grid=new_grid, axes=snapshot.axes, compositions=compositions, time=snapshot.time,
                    metadata=snapshot.metadata, source_id=snapshot.source_id, provenance=snapshot.provenance+[operation])


def interpolate_series(series, time, *, target_grid, coordinate, field_policies, mass_source):
    snapshots = list(series.snapshots)
    times = np.array([np.nan if s.time is None else s.time for s in snapshots])
    if not np.isfinite(time) or not np.all(np.isfinite(times)) or not len(times) or np.any(np.diff(times) <= 0):
        raise DataError("Time interpolation requires known, strictly increasing times")
    if time < times[0] or time > times[-1]:
        raise DataError("Time extrapolation is not supported")
    if not isinstance(target_grid, RadialGrid):
        raise DataError("Supply an explicit RadialGrid for time interpolation")
    exact = np.flatnonzero(times == time)
    i = int(exact[0]) if len(exact) else int(np.searchsorted(times, time)-1)
    j = i if len(exact) else i+1
    target = target_grid.radius_faces.read() if coordinate == "radius" else (
        target_grid.mass_faces.read() if target_grid.mass_faces is not None else None)
    if target is None:
        raise DataError("Target grid has no enclosed-mass boundaries")
    a, b = [snapshots[k].remap(target, coordinate=coordinate, mass_source=mass_source,
                             field_policies=field_policies) for k in (i, j)]
    weight = 0. if i == j else (time-times[i])/(times[j]-times[i])
    blend = lambda x, y: (1-weight)*x+weight*y
    if a.compositions.keys() != b.compositions.keys():
        raise DataError("Time interpolation requires matching composition tables")
    ma = a.derive("cell_mass", source="density").read()
    mb = b.derive("cell_mass", source="density").read()
    mass = blend(ma, mb)
    if coordinate == "mass":
        volume = blend(a.grid.volumes(), b.grid.volumes())
        r0 = blend(a.grid.radius_faces.read(0), b.grid.radius_faces.read(0))
        radius = np.cbrt(r0**3 + 3/(4*np.pi)*np.r_[0, np.cumsum(volume)])
        grid = RadialGrid(radius, mass_faces=target, frame=target_grid.frame, registry=target_grid.registry)
    else:
        grid = RadialGrid(target, radius_cells=target_grid.radius_cells,
                          frame=target_grid.frame, registry=target_grid.registry)
        if a.grid.mass_faces is not None and b.grid.mass_faces is not None:
            m0 = blend(a.grid.mass_faces.read(0), b.grid.mass_faces.read(0))
            grid = RadialGrid(target, radius_cells=target_grid.radius_cells, mass_faces=m0+np.r_[0, np.cumsum(mass)],
                              frame=target_grid.frame, registry=target_grid.registry)
    fields = [Field("density", mass/grid.volumes(), "g/cm^3", dimensions=("cell",), location="cell", sampling="average")]
    for key, f in a.fields.items():
        if key == ("cell", "density"):
            continue
        other = b.fields[key]
        if f.unit != other.unit or f.dimensions != other.dimensions or f.sampling != other.sampling:
            raise DataError("Time-interpolated fields have incompatible semantics")
        policy = field_policies.get(key, field_policies.get(f.name))
        av, bv = f.require_valid(), other.require_valid()
        if policy in ("mass_weighted", "volume_weighted"):
            wa = ma if policy == "mass_weighted" else a.grid.volumes()
            wb = mb if policy == "mass_weighted" else b.grid.volumes()
            shape = (-1,)+(1,)*(av.ndim-1)
            denominator = blend(wa, wb).reshape(shape)
            numerator = blend(av*wa.reshape(shape), bv*wb.reshape(shape))
            values = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator>0)
            valid = np.broadcast_to(denominator>0, values.shape)
        else:
            values, valid = blend(av, bv), None
        fields.append(f.replaced(values, valid=valid))
    compositions = {}
    for name, c in a.compositions.items():
        other = b.compositions[name]
        if c.species != other.species or c.basis != other.basis or c.complete != other.complete or c.interpretation != other.interpretation:
            raise DataError("Time interpolation requires identical species and abundance conventions")
        xa, xb = c.mass_fractions(), other.mass_fractions()
        # Invalid abundances in genuinely empty cells contribute zero species mass.
        def weighted(x, m):
            values, valid = x.read(), x.read_validity()
            if np.any((~valid | ~np.isfinite(values)) & (m[:, None] > 0)):
                raise DataError("Missing composition in a nonempty cell")
            return np.where(m[:, None]>0, values*m[:, None], 0.)
        total = blend(weighted(xa, ma), weighted(xb, mb))
        values = np.divide(total, mass[:, None], out=np.full_like(total, np.nan), where=mass[:, None]>0)
        if c.basis == "number_per_baryon":
            values /= np.array([s.A for s in c.species])
        compositions[name] = Composition(c.abundances.replaced(values, valid=np.broadcast_to(mass[:, None]>0, values.shape)),
            c.species, basis=c.basis, complete=c.complete, interpretation=c.interpretation, metadata=c.metadata)
    return Snapshot(fields, grid=grid, axes=a.axes, compositions=compositions, time=time,
                    metadata=a.metadata, provenance=a.provenance+b.provenance+[
                        {"operation": "time_interpolation", "bracket_s": [float(times[i]), float(times[j])], "weight": float(weight)}])

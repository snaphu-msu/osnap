# Composition Fitter Tools

This directory contains tools for reading Sukhbold progenitor profiles, repairing
reduced-network compositions, fitting Heger02-based compositions, and making
comparison plots. The newer reduced-network repair path is independent of the
Heger02 fitter.

## Ye-Repaired Reduced-Network Composition

Use `composition_fitter.reduced_composition` when you want to keep the reduced
Sukhbold progenitor composition as the starting point, but enforce consistency
with the progenitor file's direct `Ye` column.

The repair does the following for each zone:

1. Canonicalizes labels:
   - `neut`, `neutrons`, `n` become `nt1`
   - `prot`, `p` become `h1`
   - `fe` is treated as `fe56`
2. Merges any `fe` bucket into `fe56`.
3. Sets selected unstable isotopes to zero.
4. Solves a nonnegative constrained projection so the repaired abundances satisfy:
   - `sum(X_i) = 1`
   - `sum(X_i * Z_i/A_i) = Ye_column`

The default unstable isotopes are:

```python
("ti44", "cr48", "fe52", "ni56")
```

To keep `ni56` active, pass:

```python
unstable_isotopes=("ti44", "cr48", "fe52")
```

## Repair a Sukhbold Profile

```python
from composition_fitter.reduced_composition import repair_sukhbold_profile
from composition_fitter.sukhbold_profile import read_sukhbold_profile

profile = read_sukhbold_profile("progenitors/sukhbold_2016/s12.0_presn")
result = repair_sukhbold_profile(profile)

labels = result.labels
mass_fractions = result.mass_fractions
target_ye = result.target_ye
diagnostics = result.diagnostics
```

The repaired mass fractions are stored in `result.mass_fractions` with columns
given by `result.labels`.

Useful diagnostics include:

- `original_sum`
- `edited_sum_before_projection`
- `original_composition_ye`
- `edited_composition_ye`
- `repaired_composition_ye`
- `sum_error`
- `ye_error`
- `zeroed_unstable_mass`
- `merged_fe_mass`
- `projection_l2_delta`
- `optimizer_success`

## Repair Raw Arrays

Use `repair_reduced_abundances` for array-level workflows:

```python
import numpy as np

from composition_fitter.reduced_composition import repair_reduced_abundances

labels = ["neut", "h1", "he4", "ti44", "fe56", "fe"]
abundances = np.array([[0.05, 0.15, 0.30, 0.10, 0.20, 0.20]])
target_ye = np.array([0.50])

result = repair_reduced_abundances(labels, abundances, target_ye)
```

This returns a `RepairedComposition` object. In this example, the output labels
are canonicalized to `["nt1", "h1", "he4", "ti44", "fe56"]`, the `fe` mass is
merged into `fe56`, and `ti44` is zeroed before projection.

## Repair a Loaded OSNAP Progenitor

Use `osnap.reduced_composition` after loading a Kepler or MESA progenitor model:

```python
from osnap.load_data import load_kepler_progenitor
from osnap.reduced_composition import replace_progenitor_reduced_composition

prog = load_kepler_progenitor(source="sukhbold_2016", mass=12.0)
prog = replace_progenitor_reduced_composition(prog)
```

The returned `prog` has:

- `prog["profiles"]` with repaired composition columns
- `prog["nuclear_network"]` updated to the canonical repaired network
- `prog["original_nuclear_network"]` preserving the input network labels
- `prog["reduced_composition_diagnostics"]` with per-zone repair diagnostics

This function uses direct `Ye` columns only. It does not derive target `Ye` from
composition columns, and it does not modify STIR composition data.

To keep `ni56` active:

```python
prog = replace_progenitor_reduced_composition(
    prog,
    unstable_isotopes=("ti44", "cr48", "fe52"),
)
```

## Generate Validation Plots

The plotting CLI repairs one or more Sukhbold profiles and writes validation
figures:

```bash
MPLBACKEND=Agg .venv/bin/python -m composition_fitter.reduced_composition_plot
```

Default outputs are written to:

```text
output/reduced_composition_repair/
```

The default profile set includes `progenitors/sukhbold_2016/s12.0_presn`.

For a single 12.0 Msun model:

```bash
MPLBACKEND=Agg .venv/bin/python -m composition_fitter.reduced_composition_plot \
  --profiles progenitors/sukhbold_2016/s12.0_presn
```

To keep `ni56` active in the plots:

```bash
MPLBACKEND=Agg .venv/bin/python -m composition_fitter.reduced_composition_plot \
  --profiles progenitors/sukhbold_2016/s12.0_presn \
  --unstable-isotopes ti44 cr48 fe52
```

Generated plot types:

- `reduced_composition_ye_repair.png`: file-column `Ye`, original composition
  `Ye`, repaired composition `Ye`, and residuals.
- `<model>_abundance_repair.png`: selected original and repaired isotope mass
  fractions.
- `<model>_repair_diagnostics.png`: zeroed unstable mass, projection size, and
  absolute `Ye` residual.

## Validation

Run the tests with:

```bash
.venv/bin/python -m pytest tests/test_composition_fitter.py tests/test_reduced_composition.py
```

The relevant tests check:

- `fe` is merged into `fe56`
- configured unstable isotopes are zeroed
- repaired rows satisfy `sum(X)=1`
- repaired rows match the direct file-column `Ye`
- the OSNAP progenitor wrapper updates only progenitor composition data
- the validation plotting helpers create the expected axes

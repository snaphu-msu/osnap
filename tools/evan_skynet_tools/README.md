# SkyNet Tools 

## 1. **nucleo_helpers.py**

**Purpose:**  
Provides core helper functions for nucleosynthesis calculations, including interpolation, trajectory evolution, initial abundance setup, and SkyNet integration.

**Key Functions:**
- **Grid and Data Handling:**
  - `get_level(ds, dds, nblockx, blocksize)`: Computes AMR grid level.
  - `get_starting_block(dsstart, nblockx)`: Identifies level 0 blocks in a dataset.
  - `cascade_to_leaf(dsstart, point_pos, ...)`: Finds the leaf grid containing a point.
  - `shift_data(data, CG0v, CG1v, reflect)`: Shifts and reflects data arrays for boundary conditions.
  - `setup_CGs(grid, posindices, ...)`: Sets up covering grids and interpolators for physical quantities (velocity, temperature, density, etc.).

- **Trajectory Evolution:**
  - `evolve_back_one_file_many(dsstart, dsend, starting_points, ...)`: Evolves a set of points backward in time through two datasets, interpolating physical quantities along the way.

- **Model Loading:**
  - `load_kepler_model_SWH2018_solar(filename, rows_to_header)`: Loads a Kepler model.
  - `load_mesa_model_tardis(filename, rows_to_header)`: Loads a MESA model.

- **Initial Abundance Setup:**
  - `skynet_get_initY(...)`: Computes initial abundances for SkyNet from a Kepler model.
  - `skynet_get_initY_mesa(...)`: Computes initial abundances for SkyNet from a MESA model.

- **SkyNet Integration:**
  - `run_skynet(do_inv, do_screen, tfinal, trajectory_data, outfile, do_NSE, initY)`: Runs SkyNet nucleosynthesis for a given trajectory, with options for NSE or custom initial abundances.

- **Data Combination and Output:**
  - `combine_ID_with_PE(ID, PE, radius_to_join, vars)`: Combines two datasets at a given radius.
  - `print_flash_ID_file(data, filename, vars)`: Outputs combined data in a FLASH-compatible format.

- **Constants:**
  - `skynetA`, `skynetZ`, `element_list`: Lists of mass numbers, atomic numbers, and element symbols for isotopes tracked by SkyNet.

---

## 2. **do_nucleosynthesis.py**

**Purpose:**  
Main script to compute nucleosynthesis yields by tracing trajectories backward through STIR simulation outputs and running SkyNet on each trajectory.

**Key Steps:**
- **Argument Parsing:**  
  Accepts model name, MESA profile, and a flag for using plt files.

- **Setup:**  
  Loads the MESA model, sets up grid parameters, and determines when the shock reaches a target radius.

- **Trajectory Handling:**  
  Loads or computes trajectories for a set of radial points, evolving them backward in time using `nucleo_helpers.evolve_back_one_file_many`.

- **SkyNet Execution:**  
  For each trajectory, determines if NSE applies or computes initial abundances from the MESA model, then runs SkyNet.

- **Output:**  
  Saves final mass fractions and total masses for each isotope, as well as the computed trajectories.

---

## 3. **do_trajectories.py**

**Purpose:**  
Computes and saves the backward-evolved thermodynamic trajectories for a set of radial points in the STIR simulation.

**Key Steps:**
- **Argument Parsing:**  
  Accepts model name and plt flag.

- **Setup:**  
  Loads the relevant simulation output and determines the time when the shock reaches the target radius.

- **Trajectory Evolution:**  
  For each radial point, initializes a starting point and evolves it backward in time using `nucleo_helpers.evolve_back_one_file_many`, saving the full trajectory.

- **Output:**  
  Saves all trajectories to a `.npy` file for later use by do_nucleosynthesis.py.

---

## 4. **do_new_plt.py**

**Purpose:**  
Augments a STIR simulation output file (plt/chk) with nucleosynthesis yields, interpolating between the simulation and SkyNet results.

**Key Steps:**
- **Argument Parsing:**  
  Accepts model name, MESA profile, and plt flag.

- **Setup:**  
  Loads the MESA model, nucleosynthesis yields, and the relevant simulation output file.

- **Isotope Indexing:**  
  Determines which indices in the SkyNet output correspond to which isotopes in the MESA header.

- **Block Processing:**  
  For each leaf block in the simulation output:
  - Interpolates nucleosynthesis yields onto the block's radial grid.
  - For radii outside the nucleosynthesis grid, uses MESA abundances or sets simple proton/neutron values.
  - Normalizes abundances and writes them into new datasets in the HDF5 file.

- **Output:**  
  Produces a new plt/chk file with nucleosynthesis yields included as additional fields.

---

## 5. **read_prog.py**

**Purpose:**  
Converts a MESA profile file into a format compatible with FLASH simulations.

**Key Steps:**
- **Argument Parsing:**  
  Accepts a MESA profile filename.

- **Data Extraction:**  
  Reads the profile, extracts columns for radius, velocity, density, temperature, electron fraction, and mean atomic mass.

- **Unit Conversion:**  
  Converts radii to CGS, exponentiates log quantities.

- **Profile Adjustment:**  
  Flips arrays to match FLASH conventions, computes shell-averaged radii.

- **Output:**  
  Prints the processed data in a format suitable for FLASH input.

---

## **General Workflow/Integration**

1. **Trajectory Calculation:**  
   - do_trajectories.py computes backward-evolved thermodynamic trajectories for a set of radial points in the STIR simulation.

2. **Nucleosynthesis Calculation:**  
   - do_nucleosynthesis.py loads these trajectories, computes initial abundances (from MESA or Kepler), and runs SkyNet for each trajectory to get final yields.

3. **Yield Mapping:**  
   - do_new_plt.py takes the SkyNet yields and maps them onto the simulation output, producing a new plt/chk file with nucleosynthesis fields.

4. **Helper Functions:**  
   - nucleo_helpers.py provides all the core routines for interpolation, trajectory evolution, initial abundance setup, and SkyNet integration.
   - read_prog.py is a utility for converting MESA profiles to FLASH input format.

---

## **Summary Table**

| File                  | Main Purpose                                        | Key Functions/Classes                                  |
| --------------------- | --------------------------------------------------- | ------------------------------------------------------ |
| nucleo_helpers.py     | Core helpers for nucleosynthesis and SkyNet         | Interpolation, trajectory evolution, SkyNet interface  |
| do_nucleosynthesis.py | Compute nucleosynthesis yields for trajectories     | Main script, SkyNet runs, output mass fractions        |
| do_trajectories.py    | Compute backward-evolved thermodynamic trajectories | Main script, trajectory evolution, output trajectories |
| do_new_plt.py         | Map nucleosynthesis yields onto simulation output   | Main script, HDF5 manipulation, yield interpolation    |
| read_prog.py          | Convert MESA profile to FLASH input format          | Main script, data extraction, unit conversion, output  |

---

**In summary:**  
The skynet_tools directory provides a pipeline for extracting thermodynamic trajectories from STIR simulations, running nucleosynthesis calculations with SkyNet, and mapping the resulting yields back onto the simulation outputs for further analysis or use in other codes. The code is modular, with nucleo_helpers.py providing the core computational routines used by the main scripts.
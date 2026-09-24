## Component Structure

OSNAP is meant to assist with every part of the process for simulating a core-collapse supernovae, compiling observables, and analyzing the results.

However, it is modular. Most components reflect a stage and can be used individually, or as part of the larger process, or they can be skipped entirely. Each component is also flexible enough to allow easy addition of new methods and support for new pieces of software or data types.

Each component is detailed in the sections below.

### OSNAP Database (odb)

`osnap.odb` is the OSNAP database (ODB), the common scientific object model for
sharing grids, fields, composition, tracer histories, and nonspatial products
between components. It supports in-memory operations and optional HDF5 persistence,
with explicit units, metadata, validation, interpolation, and conservative remapping.
It is independent of the legacy global configuration and physics dependencies.
See the [ODB developer guide](osnap/odb/README.md) for the implemented API and
native-format support. Migrating existing workflows onto ODB is a separate task.

### Progenitor Evolution (stellar.py)
Assists in process of setup, configuration, and running of simulations for creating progenitor stars. 

Currently only planning to support MESA.

### Core Collapse & Shock Propagation (supernova.py)
Assists in setup, configuration, and running of core-collapse simulations. Currently only planning to support STIR.

The progenitor will be loaded and, if necessary, converted into a format usable as an input for STIR.

**Q: Should we support just running MESA all the way and skipping STIR? I think that defeats the purpose of the physically-motivated pipeline right?** Yes, I think so. I.e., thermal bombs with MESA. 

### Processing Simulation Results (progenitor.py & preprocess.py)
Loads in progenitor models and simulation results for use with later components. Will be capable of loading at least Kepler and MESA progenitors, as well as STIR results.

Other stellar evolution model data types (codes):
- GENEC (Hirschi)
- Limongi & Chieffi 
- Japanese code? 

### Generating Trajectories (trajectories.py)
Creates trajectories from simulation results. 

**Note: Will use a lot of modified code from flashbang.**

### Post-processing Nucleosynthesis (nucleo.py)
Uses the trajectories and progenitor composition to create trajectories.

It will calculate any needed data that is missing and ensure everything is in the same units.

Will support at least SkyNet, WinNet, pynucastro.

**Possibly: It will (optionally) pre-process the composition. If the progenitor composition is using a small network, it can put each isobar into it's most common isotopes.**

### Compiling Results (stitching.py)
Creates a data file containing the final state of the system. If the shock has not propagated through the entire star, then the progenitor data will be stitched on outside the simulation domain. Includes full ejecta composition.

### Generating Light Curves (light_curves.py)
Assists in generating light curves using SNEC, Athelas and/or STELLA.

https://github.com/athelas-astro/athelas

### Generating Spectra (spectra.py)
Assists in generating spectra using TARDIS.

Potential compatibility with:
- SEDONA 
- CMFGEN (?)

### Additional Analysis Tools (plot.py)
Tools for creating a variety of plots to analyze the resulting data. 

### Utilities (utilities.py)
Utility functions that are used by multiple components will be in this script. 

### Configuration (config.py)
Loads in configuration from a YAML file. There will be a large variety of settings, but only the ones that are non-default need to be included in the YAML.

### Setup (setup.py)
This script helps with installation and setup of OSNAP, as well as the software it uses, including MESA, STIR, SkyNet, WinNet, etc.

## Coding Philosophy

The main concerns are:
* OSNAP will be open-source.
* It is meant to be modular. Users may want to run the entire process, or they may have progenitors already ready to go, or they may only need light curves and spectra. 
* Should be as easy for users to setup and use as possible with as much of the configuration and setup being centralized as can be.

## User Experience

### Initial Setup
The setup process will go as follows:
1. Install OSNAP using PIP or Conda.
2. From command-line, go to the folder where they want things installed and run `osnap install`. This will run them through an installation process with a variety of questions about which pieces of software they plan to use, automatically downloading and doing setup for each.

### Projects
Projects are meant to have a single configuration for how models should be run. One project can be used to run any number of models, but they will all be run with the same general settings.

#### Setup
Each time the user wants to start a new project with different settings, they can either copy a previous project and tweak, or they can start a new project by running the `osnap create` command in the folder where they'd like the project to be stored. This will create a copy of the template project that comes with OSNAP. 

#### Configuration
Each project folder contains its own `config.yaml` file in which they will set how the process should be run, which components they will be using, which pieces of software they want to use, the paths to existing files, where results should be stored, and more.

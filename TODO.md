# The Refactor
Things that need to be done ASAP to make the codebase far more workable.

* ~~Create new branch backup and clear out master branch to start fresh.~~
* Data Format
  * Design abstracted data format/specification with information about units, location of data (cell center vs edge), metadata, etc.
  * Create a reader and writer for the OSNAP data format, using HDF5.
  * Implement ability for Kepler progenitors to be read into OSNAP data format.
  * Implement ability for STIR results to be read into OSNAP data format.
* Trajectories
  * Implement loading for existing trajectory data (such as from Hermansen).
  * Implement creating trajectories from STIR checkpoints.
* Nucleosynthesis
  * Implement running Nucleosynthesis using SkyNet.
* Projects
  * Implement basic setup of projects, as well as configuring and running them.

# Before Open-Source Public Release
Things that should be done before OSNAP is released to the public, roughly in order from most important to least important.

* Add metadata to the stiched results, including PNS mass and radius, explosion energy, etc so that they don't need to be recalculated during analysis.
* Implement the tools to assist in running MESA for creating progenitors.
* Implement the tools to assist in running STIR for simulating core collapse and shock propagation.
* Implement the tools for converting MESA progenitors into STIR-readable inputs.
* Add tests for every existing feature.
* Implement scripts for initial setup and project setup, including downloading and installing supported software.

# Important Features
Things that need to be done at some point but don't necessarily need to be done before open source release.

* Add support for TARDIS.
* Add support for SNEC.
* Add support for STELLA.
* Add supporf for WinNet.
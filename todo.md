# The Refactor
Things that need to be done ASAP to make the codebase far more workable.

* **Remove Dependencies:** Remove need for nucleosynth, flashbang, and progs by supporting their features natively. Will likely directly use some code from each.
* **Restructure:** Clean up existing code and reorganize it to match the new design.
* **Simplify Data:** Make it so that any existing data can be loaded easily from anywhere rather than needing to all be in the same folder as the code.
* **Implement Scripts:** Take code from many files in the scripts folder and make them an actual part of OSNAP, also ensuring they're versatile.
* **Make it easy to load existing trajectories, like those from Hermansen.
* Abstract Data Specification
* **Clean-Up:** Remove outdated or unnecessary files and code.

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
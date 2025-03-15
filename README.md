# OSNAP: One-dimensional SuperNova Analysis Pipeline

## Notes for the Tardis-Connector 2025 meetup

I created this with the goal of creating all parts of a pipeline before learning of the meetup, which is why I've given it this name. I think this could be a good repository to put our combined work into, but currently it only covers the STIR-to-MESA portion of the pipeline. It also has some code for reading MESA progenitors, which may be useful in other parts of the pipeline.

Currently, it is capable of reading in Kepler or MESA progenitors along with STIR checkpoint files, combining the data, and writing a new MESA model to a stir_output.mod file. This involves calculating some missing variables from either progenitors or the STIR output, converting STIR output to a lagrangian coordinate system, ensuring all variables are being taken from the same part of each cell, and more. It also plots the profiles for many of the non-composition variables so that we can easily spot any issues.

**However, this is still very work in progress...**

* While it outputs a MESA model which could be used, it is missing some data which needs to be calculated based on existing data. The STIR domain is missing luminosity, dq, u (cell-center riemann velocity), and mlt_vc. Kepler is also missing all of those except luminosity.
* A work-in-progress determination of the PNS boundary is currently commented out.
* While MESA model files can be read as progenitors, it is currently untested since I didn't have a STIR checkpoint that had used a MESA progenitor. It's definitely missing total specific energy and will result in an error if used without that.
* STIR has no nuclear network, so all composition is currently given by the progenitor. This is inaccurate, as composition will definitely have changed during the STIR simulation. STIR2 has a nuclear network, but it's not quite ready for use.
* For composition, the kepler progenitor is missing prot (maybe same as h1), cr60, and co56.
* The header and footer of the MESA model file have many variables not filled in, but it's unclear which ones (if any) are necessary for MESA to read/run the model file properly.

**If we choose to use this repository for the full pipeline...**

* The model_template folder was meant to be where all files necessary to run a model are kept, with all the inlists set up for a particular problem. As necessary, users would copy this folder, put their own progenitor in, run MESA, run some mesa_to_stir script, run STIR, run stir_to_mesa, and so on. Currently this template folder is just a pared down copy of the ccsn_IIp test suite folder that comes with MESA. I'm certain this process and folder structure can be simplified a lot.
* The progs folder is [a package called progs written by Zac Johnston](https://github.com/zacjohnston/progs) used for loading the kepler progenitors. There is currently no license for it but I'll want to credit him somehow if it's used.

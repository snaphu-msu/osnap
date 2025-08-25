# OSNAP: One-dimensional SuperNovae Analysis Pipeline

This is one piece to a one-dimensional supernovae simulation pipeline. It handles the stages between STIR completion and MESA shock propagation by doing the following:

1. Reads in a MESA or Kepler progenitor. (Reading in a Kepler progenitor is a work in progress)
2. Reads in STIR checkpoint data.
3. Runs SkyNet to post-process the nucleosynthesis for the STIR output. (Work in progress)
4. Stitches the progenitor data onto the STIR domain for a complete model.
5. Ensures all variables use the MESA units and are calculated for the same positions as MESA (such as cell-center vs edge velocities).
6. Creates a correctly-formatted .mod file to be read into MESA for shock propagation.

If you have any questions, email John Delker at jdelker@msu.edu.
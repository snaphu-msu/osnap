import argparse
import yt
import numpy as np
import nucleo_helpers as nhelp
import sys
import os
import h5py
import importlib

importlib.reload(nhelp)

# this script backwards traces the plt/chk files from a stir run to
# determine the nucleosynthesis

# the model was generated with the following format
# stir2_mesa_15_0.8_20_alpha1.25_hdf5_chk_????

# expect something like stir2_mesa_15_0.8_20_alpha1.25
# in the stir run directory there should be an output directory
# in the output directory there should be chk files of the form
# ./{model}_hdf5_chk_****
# the stir output should be in ./{model}.dat


def main():
    parser = argparse.ArgumentParser(
        description="Compute nucleosynthesis trajectories"
    )
    parser.add_argument("model", type=str, help="STIR model name")
    parser.add_argument(
        "-p", "--plt", action="store_true", help="Use STIR plt files"
    )

    args = parser.parse_args()

    plt_file = args.plt
    ext = "_hdf5_chk_" if not plt_file else "_hdf5_plt_cnt_"

    model = args.model
    base = "./output/" + model + ext
    datfile = "./" + model + ".dat"

    # fixed parameters for nucleosynthesis
    # this code finds when the shock crosses r_shock_target
    NSEtemp = 6e9  # K
    num_points = 250
    r_start = 15e6
    r_shock_target = 1.0e9
    r_end = 1.1e9

    nblockx = 15
    blocksize = 16
    nsub = 20

    # examine data file to find end point of nucleosynthesis
    time, rshock = np.loadtxt(datfile, unpack=True, usecols=(0, 11))
    if rshock[-1] < r_shock_target:
        print("Failed Supernova")
        sys.exit()
    else:
        i = len(rshock) - 1
        while rshock[i] > r_shock_target:
            i = i - 1

    plt_cnt_start = int(float(time[i]) * 100)

    print(datfile, plt_cnt_start)
    plt_cnt_end = 0

    plt_cnt = plt_cnt_start
    ds0 = yt.load(base + str(plt_cnt).zfill(4))

    # get index for nblockx...
    # i_nbx = 1
    # while True:
    #    if "nblockx".encode("UTF-8") == ds0._handle.handle.get("integer runtime parameters")[i_nbx][1]:
    #        print(i_nbx)
    #        break
    #    else:
    #        i_nbx += 1

    i_nbx = 75
    blocksize = ds0._handle.handle.get("integer scalars")[0][1]
    nblockx = ds0._handle.handle.get("integer runtime parameters")[i_nbx][1]

    if nblockx != 15:
        print("Warning: nblockx != 15, is this right?")
    if blocksize != 16:
        print("Warning: blocksize != 16, is this right?")

    end_radii = np.logspace(np.log10(r_start), np.log10(r_end), num_points)
    print("lowest radii:", end_radii[0])
    print("highest radii:", end_radii[-1])

    GSP = list(range(num_points))
    for i in range(num_points):
        x, y, z = end_radii[i], 0.0, 0.0
        GSP[i] = np.asarray(
            [ds0.current_time.v.item(), x, y, z, 0.0, 0.0, 0.0, 0.0, float(i)]
        )

    global_starting_points = np.asarray(GSP)
    points = np.copy(global_starting_points).reshape(
        1, len(global_starting_points), 9
    )

    mintemp = 0.0

    while plt_cnt > plt_cnt_end:
        plt_cnt -= 1
        print(plt_cnt + 1, mintemp)
        ds1 = yt.load(base + str(plt_cnt).zfill(4))
        lpoints = nhelp.evolve_back_one_file_many(
            ds0, ds1, points[-1], nsub, nblockx, blocksize, NSEtemp
        )
        mintemp = np.min(lpoints.T[4])
        if mintemp > NSEtemp:
            print("Done all trajectories")
            break
        ds0 = ds1
        points = np.concatenate((points, lpoints))

    points = np.transpose(points, (1, 2, 0))

    np.save("trajectories_" + model + ".npy", points)


if __name__ == "__main__":
    main()

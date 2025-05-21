import argparse
import numpy as np
import matplotlib.pylab as plt
import nucleo_helpers as nhelp
from nucleo_helpers import skynetA, skynetZ, element_list
import importlib
import h5py
import os
import yt
import sys

importlib.reload(nhelp)

# this routine take an original mesa model, stir output at some time,
# and the nucleosynthesis yields determined at that time from
# do_nucleosynthesis.py

def is_iso(iso):
    if iso =='neut' or iso=='prot': return True
    if iso[1].isdigit():
        element = iso[0]
        mass = iso[1:]
        if mass.isdigit():
            return True
    elif iso[2].isdigit():
        element  = iso[0:2]
        mass = iso[2:]
        if mass.isdigit():
            return True

    return False


def main():
    parser = argparse.ArgumentParser(
        description="Compute nucleosynthesis trajectories"
    )
    parser.add_argument("model", type=str, help="STIR model name")
    parser.add_argument("mesa_profile", type=str, help="MESA model name")
    parser.add_argument(
        "-p", "--plt", action="store_true", help="Use STIR plt files"
    )

    args = parser.parse_args()

    plt_file = args.plt
    ext = "_hdf5_chk_" if not plt_file else "_hdf5_plt_cnt_"

    model = args.model
    base = "./output/" + model + ext
    datfile = "./" + model + ".dat"

    # original MESA model as a profile
    mesa_final_profile = args.mesa_profile

    # nucleosynthesis grid
    num_points = 250
    r_start = 15e6
    r_shock_target = 1e9
    r_end = 1.1e9
    end_radii = np.logspace(np.log10(r_start), np.log10(r_end), num_points)

    time, rshock = np.loadtxt(datfile, unpack=True, usecols=(0, 11))
    if rshock[-1] < r_shock_target:
        print("Failed Supernova")
        sys.exit()
    else:
        i = len(rshock) - 1
        while rshock[i] > r_shock_target:
            i = i - 1

    plt_cnt_start = int(float(time[i]) * 100)

    pltfilename = (
        "./output/" + model + "_hdf5_chk_" + str(plt_cnt_start).zfill(4)
    )

    print(pltfilename)

    ds = yt.load(pltfilename)
    ds.force_periodicity()

    # mesa progenitor
    prog_header, prog_data = nhelp.load_mesa_model_tardis(mesa_final_profile, 5)
    prog_radius = 10.0 ** (prog_data.T[2]) * 6.957e10

    header = prog_header.split()
    prog_compstart = header.index("neut")  # start from 1, this is 'neut'

    if prog_compstart != 81:
        print(prog_compstart, "is not 81 check if issue")

    totalcomps = np.zeros(num_points * 686).reshape(num_points, 686)
    file = "./" + model + "_comps.npy"
    comps = np.load(file)
    totalcomps[:, :] = comps[:, 1:]

    final_plt = (
        "./" + model + "_withnucleo" + ext + str(plt_cnt_start).zfill(4)
    )

    command = "cp " + pltfilename + " " + final_plt
    os.system(command)

    indices = {}
    pi=0
    while is_iso(header[prog_compstart+pi].strip()):

        iso = header[prog_compstart+pi].strip()
        index = -1
        for i in range(len(skynetA)):
            if skynetZ[i]==0:
                if iso=='neut':
                    index = i
            elif skynetZ[i]==1 and skynetA[i]==1:
                if iso=='prot':
                    index = i
                elif iso=='h1':
                    index = i
            else:
                element = element_list[skynetZ[i]-1][0]+element_list[skynetZ[i]-1][1:]
                iso_for_i = (element+str(skynetA[i])).lower()
                if iso==iso_for_i:
                    index = i
        if index >-1:
            if iso=='h1':
                indices[iso] =  -1
            else:
                indices[iso] = index
        else:
            indices[iso] = -1

        pi += 1

    indices['trac'] = -1


    h5file = h5py.File(final_plt, "r+")
    nblockx = int(h5file["integer runtime parameters"][75][1])
    nblocks = len(h5file["temp"])
    blocksize = len(h5file["temp"][0, 0, 0, :])
    nunks = len(h5file["unknown names"])
    new_nunks = nunks + 23
    UNK = h5file.create_dataset("temp_UNK3", (new_nunks, 1), dtype="S4")
    UNK[0:nunks] = h5file["unknown names"]
    count = 0
    for index in indices:
        UNK[nunks + count] = [index.encode("UTF-8")]
        count += 1

    del h5file["unknown names"]
    h5file["unknown names"] = UNK
    del h5file["temp_UNK3"]
    count = 0
    for index in indices:
        h5file.create_dataset(index, (nblocks, 1, 1, blocksize))
        count += 1

    for grid in ds.index.grids:
        if grid.Children == []:
            # leaf block, check points
            level = int(nhelp.get_level(ds, grid.dds[0], nblockx, blocksize))

            blockid = int(grid.id - grid._id_offset)
            left_edge = [grid.LeftEdge[0], grid.LeftEdge[1], grid.LeftEdge[2]]
            CG0 = ds.covering_grid(
                level=level, left_edge=left_edge, dims=[blocksize, 1, 1]
            )

            x = np.asarray(
                np.linspace(
                    left_edge[0],
                    left_edge[0] + grid.dds[0] * blocksize,
                    blocksize,
                    endpoint=False,
                )
                + grid.dds[0] * 0.5
            )

            ye = CG0["ye  "].v[:, :, :]

            tempblock = {}
            for ele in indices:
                tempblock[ele] = np.zeros(blocksize).reshape(1, 1, blocksize)
            tempblock["sumy"] = np.zeros(blocksize).reshape(1, 1, blocksize)

            for i in range(0, blocksize):
                rad = np.sqrt(x[i] ** 2)
                if rad > r_start and rad < r_end:
                    # get nucleosynthesis from skynet output

                    index = (np.where(end_radii - rad > 0))[0][0]  # above rad
                    frac1 = (rad - end_radii[index - 1]) / (
                        end_radii[index] - end_radii[index - 1]
                    )
                    frac2 = 1.0 - frac1
                    radcomps = (
                        totalcomps[index][:] * frac1
                        + totalcomps[index - 1][:] * frac2
                    )
                    psum = 0.0
                    for ele in indices:
                        if indices[ele] != -1:
                            tempblock[ele][0, 0, i] = radcomps[indices[ele]]
                            psum += radcomps[indices[ele]]

                    # renorm to one
                    for ele in indices:
                        if indices[ele] != -1:
                            tempblock[ele][0, 0, i] /= psum

                    sumy = sum(radcomps[:] / skynetA[:])
                    tempblock["sumy"][0, 0, i] = sumy
                    tempblock["trac"][0, 0, i] = 1.0 - psum

                elif rad >= r_end:
                    initY = nhelp.skynet_get_initY_mesa(
                        header,
                        prog_radius,
                        prog_data,
                        rad,
                        skynetA,
                        skynetZ,
                        prog_compstart,
                    )
                    initX = initY * skynetA
                    psum = 0.0
                    for ele in indices:
                        if indices[ele] != -1:
                            tempblock[ele][0, 0, i] = initX[indices[ele]]
                            psum += initX[indices[ele]]
                    # renorm to one
                    for ele in indices:
                        if indices[ele] != -1:
                            tempblock[ele][0, 0, i] /= psum

                    sumy = sum(initY)
                    tempblock["sumy"][0, 0, i] = sumy
                    tempblock["trac"][0, 0, i] = 1.0 - psum
                elif rad <= r_start:
                    tempblock["prot"][0, 0, i] = ye[i, 0, 0]
                    tempblock["neut"][0, 0, i] = 1.0 - ye[i, 0, 0]
                    tempblock["sumy"][0, 0, i] = 1.0

            h5file["sumy"][blockid, :, :, :] = tempblock["sumy"]
            h5file["trac"][blockid, :, :, :] = tempblock["trac"]
            for ele in indices:
                h5file[ele][blockid, :, :, :] = tempblock[ele]

    h5file.close()


if __name__ == "__main__":
    main()

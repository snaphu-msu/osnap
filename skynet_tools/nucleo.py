import yt
import numpy as np
import skynet_tools.nucleo_helpers as nhelp
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
# in the output directory there should be chk files of the form"
# ./{model}_hdf5_chk_****
# the stir output should be in ./{model}.dat

# fixed parameters for nucleosynthesis
# this code finds when the shock crosses r_shock_target
NSEtemp = 6e9  # K
num_points = 250
r_start = 15e6
r_shock_target = 1.0e9
r_end = 1.1e9

def do_trajectories(model_path, stir_model, plt_file = False):

    ext = "_hdf5_chk_" if not plt_file else "_hdf5_plt_cnt_"
    base = model_path + "/output/" + stir_model + ext
    datfile = model_path + "/" + stir_model + ".dat"

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

    np.save("trajectories_" + stir_model + ".npy", points)

def do_nucleo(model_path, stir_model, mesa_profile, plt_file = False):

    ext = "_hdf5_chk_" if not plt_file else "_hdf5_plt_cnt_"
    base = model_path + "/output/" + stir_model + ext
    datfile = model_path + "/" + stir_model + ".dat"

    # will be model specific
    prog_header, prog_data = nhelp.load_mesa_model_tardis(mesa_profile, 5)
    prog_radius = 10.0 ** (prog_data.T[2]) * 6.957e10

    header = prog_header.split()
    print(header)
    prog_compstart = header.index("neut")  # start from 1, this is 'neut'

    print(prog_compstart, "should be 81?")

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

    end_radii = np.logspace(np.log10(r_start), np.log10(r_end), num_points)
    print("lowest radii:", end_radii[0])
    print("highest radii:", end_radii[-1])
    volume = np.zeros(num_points)
    volume[0] = (4.0 * np.pi / 3.0) * (
        ((end_radii[1] + end_radii[0]) / 2.0) ** 3 - end_radii[0] ** 3
    )
    volume[-1] = (4.0 * np.pi / 3.0) * (
        end_radii[-1] ** 3 - ((end_radii[-1] + end_radii[-2]) / 2.0) ** 3
    )
    for i in range(1, num_points - 1):
        volume[i] = (4.0 * np.pi / 3.0) * (
            ((end_radii[i + 1] + end_radii[i]) / 2.0) ** 3
            - ((end_radii[i] + end_radii[i - 1]) / 2.0) ** 3
        )
    print(sum(volume))

    GSP = list(range(num_points))
    for i in range(num_points):
        x, y, z = end_radii[i], 0.0, 0.0
        GSP[i] = np.asarray(
            [ds0.current_time.v.item(), x, y, z, 0.0, 0.0, 0.0, 0.0, float(i)]
        )
        # print(GSP[i][1])

    global_starting_points = np.asarray(GSP)
    points = np.copy(global_starting_points).reshape(
        1, len(global_starting_points), 9
    )

    mintemp = 0.0

    points = np.load("trajectories_" + stir_model + ".npy")
    ncomps = 686

    fcomps = np.zeros((ncomps + 1) * num_points).reshape(
        num_points, (ncomps + 1)
    )
    for j in range(num_points):
        print(j, "of", num_points)

        trajectory_data = np.asarray(
            [
                points[j][0][:],
                points[j][4][:] / 1e9,
                points[j][5][:],
                points[j][6][:],
            ]
        ).T
        pID = int(points[j][8][0])
        trajectory_data = np.flip(trajectory_data, 0)

        tfinal = trajectory_data[-1][0]

        last_point = trajectory_data[-1] * 1.0
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        last_point[1] = 1e-5
        last_point[2] = 1e-10
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] += trajectory_data[-1][0] - trajectory_data[-2][0]
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] = 1000.0
        trajectory_data = np.concatenate((trajectory_data, [last_point]))
        last_point[0] = 2000.0
        trajectory_data = np.concatenate((trajectory_data, [last_point]))

        filebase = "./comps/tardis_final_" + stir_model + str(pID)
        #        filebase = "./temp_"+str(pID)

        # NSE or not?
        if trajectory_data[0, 1] > NSEtemp / 1e9:
            output = nhelp.run_skynet(
                True, True, tfinal, trajectory_data, outfile=filebase
            )
        else:
            starting_radius = np.sqrt(
                points[j][1][-1] ** 2
                + points[j][2][-1] ** 2
                + points[j][3][-1] ** 2
            )
            initY = nhelp.skynet_get_initY_mesa(
                header,
                prog_radius,
                prog_data,
                starting_radius,
                nhelp.skynetA,
                nhelp.skynetZ,
                prog_compstart,
            )
            output = nhelp.run_skynet(
                True,
                True,
                tfinal,
                trajectory_data,
                outfile=filebase,
                do_NSE=False,
                initY=initY,
            )

        h5file = h5py.File(filebase + ".h5", "r")

        As = h5file["A"]
        # Zs = h5file['Z']

        # steps = len(h5file['Y'])

        final_massfracs = h5file["Y"][-1] * As[:]
        #    final_massfracs2 = np.asarray(output.FinalY())[:]*skynetA[:]
        #    print(np.array_equal(final_massfracs,final_massfracs2))

        fcomps[j, 0] = float(pID)
        fcomps[j, 1:] = final_massfracs[:]

        h5file.close()
    #        os.system("rm "+filebase+".h5; rm "+filebase+".log")

    np.save("tardis_final_" + stir_model + "_comps.npy", fcomps)

    # dens = np.zeros(num_points)
    # mass = np.zeros(num_points)

    # for i in range(num_points):
    #    dens[i] = trajs[i,5,0]
    #    mass[i] = dens[i]*volume[i]

    totalmasses = {}

    for j in range(ncomps):
        if nhelp.skynetZ[j] == 0:
            iso = "neut"
        else:
            element = (
                nhelp.element_list[nhelp.skynetZ[j] - 1][0].upper()
                + nhelp.element_list[nhelp.skynetZ[j] - 1][1:]
            )
            iso = element + str(nhelp.skynetA[j])

        totalmasses[iso] = sum(volume[:] * points[:, 5, 0] * fcomps[:, 1 + j])

    np.save(stir_model + "_totalmasses.npy", totalmasses)
    print(totalmasses)
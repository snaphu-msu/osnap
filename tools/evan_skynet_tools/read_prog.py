#!/usr/bin/env python
# converts a mesa profile file to a .flash file.
import argparse
import numpy as np
import sys

Rsun = 6.957e10
Msun = 1.9884098706980504e33

f = sys.argv[1]


def main():
    parser = argparse.ArgumentParser(description="MESA to .FLASH")
    parser.add_argument("fn", type=str, help="MESA profile file")

    args = parser.parse_args()
    fn = args.fn

    with open(f, "r") as P:
        for i, line in enumerate(P):
            if i == 5:
                col = line.split()
                break

    print(f"# {fn}")

    iradius = col.index("radius")
    itemp = col.index("logT")
    idens = col.index("logRho")
    ivelr = col.index("velocity")
    iye = col.index("ye")
    iabar = col.index("abar")

    radius, velx, dens, temp, ye, abar = np.loadtxt(
        f, usecols=(iradius, ivelr, idens, itemp, iye, iabar), unpack=True, skiprows=7
    )
    radius *= Rsun
    temp = np.power(10.0, temp)
    dens = np.power(10.0, dens)

    npoints = np.size(radius)

    sumy = 1.0 / abar

    radius = np.flip(radius)
    velx = np.flip(velx)
    dens = np.flip(dens)
    temp = np.flip(temp)
    ye = np.flip(ye)
    sumy = np.flip(sumy)

    print("number of variables = 5")
    print("temp")
    print("dens")
    print("velx")
    print("ye")
    print("sumy")
    # print("magx")
    # print("magy")

    # Fix up the radius.  Radius from file is outer radius of mass shell.
    drad = []
    drad = drad + [radius[0]]
    for i in range(1, npoints):
        drad = drad + [radius[i] - radius[i - 1]]
    for i in range(0,npoints):
       radius[i] = radius[i] - 0.5*drad[i]

    # Fix up the velocity of the first zone.  All other variables we will assume are piece-wise constant.
    # dvdr = velx[0] / drad[0]
    # velx[0] = radius[0]*dvdr

    for i in range(0, npoints - 1):
        print(radius[i], temp[i], dens[i], velx[i], ye[i], sumy[i])


if __name_ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit
import time
import os, glob
import random
import config as cf

sweeps = 500
K = 1
J = 1
H = 0
lattice = cf.L
relax = 100
sample_step = 50


@jit
def wolff_cluster(P_add, spin_m, spin):
    # mag = []
    # mag_2 = []

    for s in range(sample_step):
        if s > 0:
            for idx in range(lattice):
                for idy in range(lattice):
                    if random.random() > 0.5:
                        spin[idx, idy] = -spin[idx, idy]
        for k in range(sweeps + relax):
            x = np.random.randint(0, lattice)
            y = np.random.randint(0, lattice)

            sign = spin[x, y]
            stack = [[x, y]]
            lable = [[1 for i in range(lattice)] for j in range(lattice)]
            lable[x][y] = 0

            while len(stack) > 0.5:

                # While stack is not empty, pop and flip a spin
                [x_site, y_site] = stack.pop()
                spin[x_site, y_site] = -sign

                # Append neighbor spins

                # Left neighbor
                if x_site < 0.5:
                    [leftx, lefty] = [lattice - 1, y_site]
                else:
                    [leftx, lefty] = [x_site - 1, y_site]

                if spin[leftx, lefty] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([leftx, lefty])
                    lable[leftx][lefty] = 0

                # Right neighbor
                if x_site > lattice - 1.5:
                    [rightx, righty] = [0, y_site]
                else:
                    [rightx, righty] = [x_site + 1, y_site]

                if spin[rightx, righty] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([rightx, righty])
                    lable[rightx][righty] = 0

                # Up neighbor
                if y_site < 0.5:
                    [upx, upy] = [x_site, lattice - 1]
                else:
                    [upx, upy] = [x_site, y_site - 1]

                if spin[upx, upy] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([upx, upy])
                    lable[upx][upy] = 0

                # Down neighbor
                if y_site > lattice - 1.5:
                    [downx, downy] = [x_site, 0]
                else:
                    [downx, downy] = [x_site, y_site + 1]

                if spin[downx, downy] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([downx, downy])
                    lable[downx][downy] = 0
        if (s % 5) == 0:
            print("s is:", s)
        spin_m[s,:] = spin.flatten()

    return spin_m

def main_loop(t):
    #t = T / 10
    P_add = 1 - np.exp(-2 * J / t)
    spin_m = np.ones((sample_step, lattice * lattice))
    spin = np.random.randint(-1, 1, (lattice, lattice))
    spin[spin==0] = 1
    mag = wolff_cluster(P_add, spin_m, spin)
    print("mag.shape is:", mag.shape)
    print("t is", t)
    m, n = mag.shape
    for i in range(m):
        np.save("{:d}_L/{:f}_{:d}.npy".format(lattice, t, i), np.reshape(mag[i, :], [1, -1]))


def single_run():
    start = time.time()
    for T in np.linspace(0.8, 3.6, 280):
        main_loop(T)
    end = time.time()
    print("run time is:", (end - start))


def multi_run():
    start = time.time()

    ing_argv = []
    for T in np.linspace(1.2, 3.6, 140):
        ing_argv.append(T)
    with Pool(4) as p:
        p.map(main_loop, ing_argv)

    end = time.time()

    print("run time is:", (end - start))


if __name__ == "__main__":
    multi_run()

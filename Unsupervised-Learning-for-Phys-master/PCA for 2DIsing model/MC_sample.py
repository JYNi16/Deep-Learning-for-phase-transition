import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit
import time
import os, glob

sweeps = 500
K = 1
J = 1
H = 0
lattice = [20, 40, 80]
relax = 100
sample_step = 100

def make_dir():
    for L in lattice:
        if not os.path.exists("{:d}_L".format(L)):
            os.mkdir("{:d}_L".format(L))

@jit
def wolff_cluster(t, Lattice):
    # mag = []
    # mag_2 = []
    spin_m = np.ones((sample_step, Lattice*Lattice))
    for s in range(sample_step):
        spin = np.random.randint(-1, 1, (Lattice, Lattice))
        for idx in range(Lattice):
            for idy in range(Lattice):
                if spin[idx, idy] == 0:
                    spin[idx, idy] = 1
        for k in range(sweeps + relax):
            x = np.random.randint(0, Lattice)
            y = np.random.randint(0, Lattice)
            sign = spin[x, y]
            P_add = 1 - np.exp(-2 * J / t)
            stack = [[x, y]]
            lable = [[1 for i in range(Lattice)] for j in range(Lattice)]
            lable[x][y] = 0

            while len(stack) > 0.5:

                # While stack is not empty, pop and flip a spin
                [x_site, y_site] = stack.pop()
                spin[x_site, y_site] = -sign

                # Append neighbor spins

                # Left neighbor
                if x_site < 0.5:
                    [leftx, lefty] = [Lattice - 1, y_site]
                else:
                    [leftx, lefty] = [x_site - 1, y_site]

                if spin[leftx, lefty] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([leftx, lefty])
                    lable[leftx][lefty] = 0

                # Right neighbor
                if x_site > Lattice - 1.5:
                    [rightx, righty] = [0, y_site]
                else:
                    [rightx, righty] = [x_site + 1, y_site]

                if spin[rightx, righty] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([rightx, righty])
                    lable[rightx][righty] = 0

                # Up neighbor
                if y_site < 0.5:
                    [upx, upy] = [x_site, Lattice - 1]
                else:
                    [upx, upy] = [x_site, y_site - 1]

                if spin[upx, upy] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([upx, upy])
                    lable[upx][upy] = 0

                # Down neighbor
                if y_site > Lattice - 1.5:
                    [downx, downy] = [x_site, 0]
                else:
                    [downx, downy] = [x_site, y_site + 1]

                if spin[downx, downy] * sign > 0.5 and np.random.rand() < P_add:
                    stack.append([downx, downy])
                    lable[downx][downy] = 0
        if (s % 20) == 1:
            print("s is:", s)
        spin_m[s,:] = spin.flatten()

    return spin_m

def main_loop(T):
    t = T / 10
    for Lattice in lattice:
        mag = wolff_cluster(t, Lattice)
        print("mag.shape is:", mag.shape)
        print("t is", t)
        print("Lattice is:", Lattice)
        m,n = mag.shape
        for i in range(m):
            np.save("{:d}_L/{:d}_{:d}.npy".format(Lattice, T, i), np.reshape(mag[i,:], [1,-1]))


def single_run():
    start = time.time()
    for T in range(3, 30, 1):
        main_loop(T)

    end = time.time()
    print("run time is:", (end - start))


def multi_run():
    start = time.time()

    ing_argv = []
    for T in range(10, 30, 1):
        ing_argv.append(T)
    with Pool(8) as p:
        p.map(main_loop, ing_argv)

    end = time.time()

    print("run time is:", (end - start))


if __name__ == "__main__":
    make_dir()
    multi_run()
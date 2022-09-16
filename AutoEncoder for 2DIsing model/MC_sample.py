import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from numba import jit
import time

sweeps = 500
K = 1
J = 1
H = 0
Lattice = 20
relax = 100
sample_step = 200

@jit
def wolff_cluster(t, spin_m):
    # mag = []
    # mag_2 = []
    for s in range(sample_step):
        spin = np.random.randint(-1, 1, (Lattice, Lattice))
        for x_idx in range(Lattice):
            for y_idx in range(Lattice):
                if spin[x_idx, y_idx] == 0:
                    spin[x_idx, y_idx] = 1
        #spin[spin == 0] = 1
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

def main_loop(t):
    spin_m = np.ones([sample_step, Lattice*Lattice])
    mag = wolff_cluster(t, spin_m)
    print("mag.shape is:", mag.shape)
    print("t is", t)
    m,n = mag.shape
    for i in range(m):
        np.save("{:d}_L/{:f}_{:d}.npy".format(Lattice, t, i), np.reshape(mag, [1,-1]))


def single_run():
    start = time.time()
    for T in np.linspace(1, 3.5, 125):
        main_loop(T)

    end = time.time()
    print("run time is:", (end - start))


def multi_run():
    start = time.time()

    ing_argv = []
    for T in np.linspace(1, 3.5, 125):
        main_loop(T)
    with Pool(4) as p:
        p.map(main_loop, ing_argv)

    end = time.time()

    print("run time is:", (end - start))


if __name__ == "__main__":
    multi_run()
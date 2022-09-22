#This script is coded for resconstructing order parameters by PCA

from sklearn.decomposition import PCA
import numpy as np
import glob, os, random
import matplotlib.pyplot as plt
import sklearn_pca

def pca_spin(data):
    pca = PCA(n_components=2)
    pca.fit(data)
    v = pca.components_
    return v

def load_MCspin(path):
    data_list = []
    color_z = []
    data_list.extend(glob.glob(os.path.join(path, "*_25.npy")))

    for i in range(len(data_list)):
        color_z.append(float((data_list[i].split("\\")[-1]).split("_")[0]))

    return data_list, color_z


def m_average(path1, path2, path3):
    data1, _ = sklearn_pca.load_data(path1)
    data2, _ = sklearn_pca.load_data(path2)
    data3, _ = sklearn_pca.load_data(path3)

    v1 = pca_spin(data1)
    v2 = pca_spin(data2)
    v3 = pca_spin(data3)

    data_list1, color_z = load_MCspin(path1)
    data_list2, _ = load_MCspin(path2)
    data_list3, _ = load_MCspin(path3)

    m1, m2, m3 = [], [], []
    for i in range(len(data_list1)):
        tmp_data1 = np.reshape(np.load(data_list1[i]), [-1, 1])
        m1.append(np.abs(np.dot(v1[0:1,:], tmp_data1)[0][0]/20))

        tmp_data2 = np.reshape(np.load(data_list2[i]), [-1, 1])
        m2.append(np.abs(np.dot(v2[0:1,:], tmp_data2)[0][0]/40))

        tmp_data3 = np.reshape(np.load(data_list3[i]), [-1, 1])
        m3.append(np.abs(np.dot(v3[0:1,:], tmp_data3)[0][0]/80))

    return color_z, m1, m2, m3

def plot_spin(t, m1, m2, m3):
    plt.figure(figsize=(6, 6))
    plt.plot(t, m1, marker="o", label="20")
    plt.plot(t, m2, marker="o", label="40")
    plt.plot(t, m3, marker="o", label="80")
    plt.xlabel("temperatue")
    plt.ylabel("C")
    plt.legend()
    plt.show()

if __name__=="__main__":
    path_1 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/20_L_s"
    path_2 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/40_L_s"
    path_3 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/80_L_s"
    color_z, m1, m2, m3 = m_average(path_1, path_2, path_3)
    plot_spin(color_z, m1, m2, m3)
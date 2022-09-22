from sklearn.decomposition import PCA
import numpy as np
import glob, os, random
import matplotlib.pyplot as plt

def load_data(root_path):

    data_list = []
    color_z = []
    data_list.extend(glob.glob(os.path.join(root_path, "*.npy")))
    random.shuffle(data_list)

    spin_m_data = np.load(data_list[0])

    for i in range(len(data_list)):
        color_z.append(float((data_list[i].split("\\")[-1]).split("_")[0]) / 10)

    for i in range(1, len(data_list)):
        tmp_data = np.load(data_list[i])
        spin_m_data = np.concatenate((spin_m_data, tmp_data), axis=0)

    return spin_m_data, color_z

def pca_data(data_path):
    spin_m_data, color_z = load_data(data_path)
    pca = PCA(n_components=2)
    pca.fit(spin_m_data)
    low_data = pca.transform(spin_m_data)

    return low_data, color_z

def pca_spin(data1, data2, data3, path1, path2, path3):
    data_list1 = []
    data_list2 = []
    data_list3 = []
    color_z = []
    data_list1.extend(glob.glob(os.path.join(path1, "*_50.npy")))
    data_list2.extend(glob.glob(os.path.join(path2, "*_50.npy")))
    data_list3.extend(glob.glob(os.path.join(path3, "*_50.npy")))
    pca = PCA(n_components=1)
    pca.fit(data1)
    v1 = pca.components_

    pca = PCA(n_components=1)
    pca.fit(data2)
    v2 = pca.components_

    pca = PCA(n_components=1)
    pca.fit(data3)
    v3 = pca.components_

    for i in range(len(data_list1)):
        color_z.append(float((data_list3[i].split("\\")[-1]).split("_")[0]))

    m1, m2, m3 = [], [], []
    for i in range(len(data_list1)):
        tmp_data1 = np.reshape(np.load(data_list1[i]), [-1, 1])
        m1.append(np.abs(np.dot(v1, tmp_data1)[0][0]/20))

        tmp_data2 = np.reshape(np.load(data_list2[i]), [-1, 1])
        m2.append(np.abs(np.dot(v2, tmp_data2)[0][0]/40))

        tmp_data3 = np.reshape(np.load(data_list3[i]), [-1, 1])
        m3.append(np.abs(np.dot(v3, tmp_data3)[0][0]/80))

    print(color_z, m1)
    plt.plot(color_z, m1, marker="o", label="20")
    plt.plot(color_z, m2, marker="o", label="40")
    plt.plot(color_z, m3, marker="o", label="80")
    plt.xlabel("temperatue")
    plt.ylabel("m")
    plt.legend()
    plt.show()

def plot(path1, path2, path3):
    data1, color_1 = pca_data(path1)
    data2, color_2 = pca_data(path2)
    data3, color_3 = pca_data(path3)

    plt.subplot(3,1,1)
    plt.scatter(data1[:,0], data1[:,1], c=color_1)
    plt.colorbar()
    plt.xlim(-100, 100)
    plt.ylim(-50, 50)
    plt.legend()

    plt.subplot(3,1,2)
    plt.scatter(data2[:,0], data2[:,1], c=color_2)
    plt.colorbar()
    plt.xlim(-100, 100)
    plt.ylim(-50, 50)

    plt.subplot(3,1,3)
    plt.scatter(data3[:,0], data3[:,1], c=color_3)
    plt.colorbar()
    plt.xlim(-100, 100)
    plt.ylim(-50, 50)

    plt.savefig("figure/pca.jpg")

    plt.show()



if __name__=="__main__":
    root_path_1 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/20_L_s"
    root_path_2 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/40_L_s"
    root_path_3 = "E:/deeplearning/Unsupervised learning for Phys//PCA for 2DIsing model/80_L_s"

    plot(root_path_1, root_path_2, root_path_3)
import time
from PIL import Image
import numpy as np


def PCA(num):
    path = ''
    if num in range(0, 10):
        path = "./airplane/airplane0" + str(num) + ".tif"
    if num in range(10, 100):
        path = "./airplane/airplane" + str(num) + ".tif"
    img = Image.open(path)
    data = np.asarray(img)
    data_x = np.hstack((data[:, :, 0], data[:, :, 1], data[:, :, 2]))  # 将三个颜色的向量按照列合并在一起
    data_mean = np.repeat(np.mean(data_x, axis=1).reshape((1, 256)), 768, axis=0).T  # 每列的均值，每列表示一个样本点
    data_y = data_x - data_mean  # 1.样本点去中心化
    S = np.dot(data_y, data_y.T) / (data_x.shape[1] - 1)  # 协方差矩阵
    # 对协方差矩阵进行特征值分解
    U = None
    eigValues = []
    while 1:
        itrs_num = 0
        delta = float('inf')
        N = np.shape(S)[0]
        x_old = np.ones(shape=(N, 1))
        x_new = np.ones(shape=(N, 1))
        while itrs_num < 100 and delta > 0.001:
            itrs_num += 1
            y = np.dot(S, x_old)
            x_new = y / np.linalg.norm(y)
            delta = np.linalg.norm(x_new - x_old)
            x_old = x_new
        eigVec = x_new  # 特征向量（竖版）
        eigValue = np.dot(eigVec.T, np.dot(S, eigVec))[0][0]  # 特征值
        if eigValue < 0.01:
            break
        S = S - eigValue * np.dot(eigVec, eigVec.T)  # 减去秩一矩阵
        eigValue = np.sqrt(eigValue)  # 奇异值
        if U is None:
            U = eigVec.T
        else:
            U = np.append(U, eigVec.T, axis=0)  # 特征向量（横版）
        eigValues.append(eigValue)

    sum_all = np.sum(eigValues)
    sum_k = 0
    alpha = 0.8
    k = 0
    for i in range(len(eigValues)):
        sum_k += eigValues[i]
        if sum_k / sum_all >= alpha:
            k = i
            break

    data_z = np.dot(U[:k].T, np.dot(U[:k], data_y)) + data_mean
    im1_channels = np.hsplit(data_z, 3)
    im2 = np.zeros((im1_channels[0].shape[0], im1_channels[0].shape[0], 3))
    for i in range(3):
        im2[:, :, i] = im1_channels[i]

    im2 = im2.astype('uint8')
    im3 = Image.fromarray(im2)
    im3.save("./plane_out/" + str(num) + '.tif')
    return k / 192, np.sum((data_z - data_x) ** 2)


if __name__ == '__main__':
    time_list = np.array([])
    yasuolv = np.array([])
    chonggouwucha = np.array([])
    # 计算平均耗时、压缩率、重构误差
    for j in range(0, 100):
        ts = time.time()
        y, c = PCA(j)
        te = time.time()
        yasuolv = np.append(yasuolv, y)
        time_list = np.append(time_list, te - ts)
        chonggouwucha = np.append(chonggouwucha, c)
        print(yasuolv[j])
        print(time_list[j])
        print(chonggouwucha[j])
        print("===================")

    print(yasuolv.mean())
    print(time_list.mean())
    print(chonggouwucha.mean())

import time

from PIL import Image
import numpy as np


class PCA:
    def __init__(self):
        self.path_list = []
        for i in range(0, 10):  # 调整要输入的图片个数
            self.path_list.append("./agricultural/agricultural0" + str(i) + ".tif")
        for i in range(10, 100):
            self.path_list.append("./agricultural/agricultural" + str(i) + ".tif")
        self.X=None # 三个矩阵组成的数组，原始数据
        self.Y = None  # 去中心化的数据
        self.mean = None  # 平均值
        self.S = None  # 即将被求特征值和特征向量的方阵
        self.U = None  # 特征矩阵
        self.eigValues = []  # 特征值
        self.k = 0
        self.W = None  # 取出的前k个特征向量
        self.num = 0
        self.Z=None # 最终得到的数据

    def openImg(self, num):
        self.num = num
        image_dir = self.path_list[num]
        img = Image.open(image_dir)
        data = np.asarray(img)
        data1 = np.hstack((data[:, :, 0], data[:, :, 1], data[:, :, 2]))  # 将三个颜色的向量按照列合并在一起
        self.X=data1
        data_mean = np.mean(data1, axis=0)
        data_y = data1 - data_mean  # 1.样本点去中心化
        self.Y = data_y
        self.mean = data_mean
        S = np.dot(data_y, data_y.T) / (data1.shape[0] - 1)  # 协方差矩阵
        # S1 = np.dot(S, S.T)  # 计算矩阵AAT的特征值和特征向量->U
        # self.S = S1
        self.S = S

    def eigenDecomposition(self):  # 计算奇异值分解
        mat = self.S
        while 1:
            itrs_num = 0
            delta = float('inf')
            N = np.shape(mat)[0]
            x_old = np.ones(shape=(N, 1))
            x_new = np.ones(shape=(N, 1))
            while itrs_num < 100 and delta > 0.001:
                itrs_num += 1
                y = np.dot(mat, x_old)
                m = np.amax(y)
                x_new = y / m
                delta = np.linalg.norm(x_new - x_old)
                x_old = x_new
            eigVec = x_new / np.linalg.norm(x_new)  # 特征向量（竖版）
            eigValue = np.dot(eigVec.T, np.dot(mat, eigVec))[0][0]  # 其实是奇异值
            if eigValue < 0.01:
                break
            mat = mat - eigValue * np.dot(eigVec, eigVec.T)
            eigValue = np.sqrt(eigValue)
            if self.U is None:
                self.U = eigVec.T
            else:
                self.U = np.append(self.U, eigVec.T, axis=0)  # 特征向量（横版）
            self.eigValues.append(eigValue)

    def pickK(self):
        sum_all = np.sum(self.eigValues)
        sum_k = 0
        alpha = 0.7
        k = 0
        for i in range(len(self.eigValues)):
            sum_k += self.eigValues[i]
            if sum_k / sum_all >= alpha:
                k = i
                break
        self.W = self.U[:k]
        self.k = k

    def new(self):
        im1_channels = np.hsplit(self.Y, 3)
        im1_mean = np.hsplit(self.mean, 3)
        for i in range(3):
            im1_channels[i] = np.dot(self.W.T, np.dot(self.W, im1_channels[i])) + im1_mean[i]
        self.Z=np.hstack((im1_channels[0],im1_channels[1],im1_channels[2]))
        im2 = np.zeros((im1_channels[0].shape[0], im1_channels[0].shape[0], 3))
        for i in range(3):
            im2[:, :, i] = im1_channels[i]
        im2 = im2.astype('uint8')
        im3 = Image.fromarray(im2)
        im3.save('./out/' + str(self.num) + '.tif')

    def clear(self):
        self.U = None
        self.eigValues=[]


if __name__ == '__main__':
    PCA1 = PCA()
    time_list = np.array([])
    yasuolv = np.array([])
    chonggouwucha=np.array([])

    # 计算平均耗时、压缩率、重构误差
    for j in range(0, 100):
        ts = time.time()
        PCA1.openImg(j)
        PCA1.eigenDecomposition()
        PCA1.pickK()
        PCA1.new()
        PCA1.clear()
        te = time.time()

        time_list = np.append(time_list, te - ts)
        yasuolv = np.append(yasuolv, PCA1.k / 192)
        a=PCA1.X-PCA1.Z
        chonggouwucha=np.append(chonggouwucha,np.sum(a ** 2))
        print(yasuolv[j])
        print(time_list[j])
        print(chonggouwucha[j])

    print(yasuolv)
    print(time_list)
    print(chonggouwucha)

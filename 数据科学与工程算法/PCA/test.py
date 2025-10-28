import time

import numpy as np
from PIL import Image


def frobenius_norm(M):
    return np.linalg.norm(M, ord='fro')


def shrink(M, tau):
    return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))


def svd_threshold(M, tau):
    U, S, V = np.linalg.svd(M, full_matrices=False)
    return np.dot(U, np.dot(np.diag(shrink(S, tau)), V))


class R_pca:

    def __init__(self, D):

        self.D = D  # 传入的矩阵
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)
        self.L = np.zeros(self.D.shape)

        self.mu = np.prod(self.D.shape) / (4 * np.linalg.norm(self.D, ord=1))

        self.mu_inv = 1 / self.mu

        self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    def fit(self, max_iter=1000):
        num = 0
        err = float('inf')
        Sk = self.S
        Yk = self.Y
        Lk = self.L
        _tol = 1E-7 * frobenius_norm(self.D)
        ts=time.time()
        while (err > _tol) and num < max_iter:
            Lk = svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)  # step 3
            Sk = shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)  # step 4
            Yk = Yk + self.mu * (self.D - Lk - Sk)  # step 5
            err = frobenius_norm(self.D - Lk - Sk)  # F范数损失
            num += 1
            if num%50==0:
                te=time.time()
                print('iteration: {0}, error: {1}, time: {2}'.format(num, err,te-ts))
        return Lk, Sk


img = Image.open("./airplane/airplane31.tif")
data = np.asarray(img)
D = np.hstack((data[:, :, 0], data[:, :, 1], data[:, :, 2]))  # 将三个颜色的向量按照列合并在一起

rpca = R_pca(D)
L, S = rpca.fit(max_iter=300)

im1_channels = np.hsplit(L, 3)
im2 = np.zeros((256, 256, 3))
for i in range(3):
    im2[:, :, i] = im1_channels[i]
im2 = im2.astype('uint8')
im3 = Image.fromarray(im2)
im3.save('31L.tif')

im1_channels = np.hsplit(S, 3)
im2 = np.zeros((256, 256, 3))
for i in range(3):
    im2[:, :, i] = im1_channels[i]
im2 = im2.astype('uint8')
im3 = Image.fromarray(im2)
im3.save('31S.tif')

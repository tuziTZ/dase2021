import csv

import numpy as np
import random


def nmf(X, r, maxiter, minError):
    # X=U*V'
    row, col = X.shape
    U = np.around(np.array(np.random.rand(row, r)), 5)
    V = np.around(np.array(np.random.rand(col, r)), 5)
    obj = []
    for iter in range(maxiter):
        print('-----------------------------')
        print('开始第', iter, '次迭代')
        # update U
        XV = np.dot(X, V)
        UVV = np.dot(U, np.dot(V.T, V))
        U = (U * (XV / np.maximum(UVV, 1e-10)))
        # update V
        XU = np.dot(X.T, U)
        VUU = np.dot(V, np.dot(V.T, V))
        V = (V * (XU / np.maximum(VUU, 1e-10)))
        d = np.diag(1 / np.maximum(np.sqrt(np.sum(V * V, 0)), 1e-10))
        V = np.dot(V, d)

        temp = X - np.dot(U, np.transpose(V))
        error = np.sum(temp * temp)
        print('error:', error)
        print('第', iter, '次迭代结束')
        obj.append(error)
        if error < minError:
            break
    return U, V, obj


if __name__ == "__main__":
    with open('ratings.csv', 'r') as file:
        reader = csv.reader(file)
        x = list(reader)
    data = np.array(x[1:], dtype=float)
    max_list = list(np.max(data, axis=0))
    n = max_list[0]
    m = max_list[1]
    X = np.zeros((int(n), int(m)))
    for row in x[1:]:
        X[int(row[0]) - 1][int(row[1]) - 1] = float(row[2])
    # print('X:',X)
    U, V, obj = nmf(X, 2, 100, 0.01)
    new_X=np.dot(U,V.T)
    for row in new_X:
        print(row)
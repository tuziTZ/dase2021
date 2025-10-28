import csv
import time

import numpy as np
import random


def NMF(V, K=10, times=500):
    row, col = V.shape
    W = np.random.rand(row, K)
    H = np.random.rand(col, K)
    rmse_list=[]
    mae_list=[]
    for t in range(times):
        print('-----------------------------')
        print('开始第', t, '次迭代')
        ts=time.time()
        # update W
        XH = np.dot(V, H)
        WHH = np.dot(W, np.dot(H.T, H))
        W = (W * (XH / np.maximum(WHH, 1e-10)))
        # update H
        XW = np.dot(V.T, W)
        HWW = np.dot(H, np.dot(H.T, H))
        H = (H * (XW / np.maximum(HWW, 1e-10)))
        d = np.diag(1 / np.maximum(np.sqrt(np.sum(H * H, 0)), 1e-10))
        H = np.dot(H, d)
        te = time.time()
        print(te - ts)
    V_new = np.dot(W, np.transpose(H))
    #     rmse = 0
    #     mae = 0
    #     cnt = 0
    #     for i in range(row):
    #         for j in range(col):
    #             if V[i][j] > 0:
    #                 rmse += pow((V[i][j] - V_new[i][j]), 2)
    #                 mae += abs(V[i][j] - V_new[i][j])
    #                 cnt += 1
    #     rmse = np.sqrt(rmse / cnt)
    #     mae = mae / cnt
    #     print(' rmse=' + str(rmse) + ' mae=' + str(mae))
    #     rmse_list.append(rmse)
    #     mae_list.append(mae)
    # print(rmse_list)
    # print(mae_list)

    return V_new


if __name__ == "__main__":
    query_num = 6
    x=[]
    with open("./ml-1m/ratings.dat", 'r') as file:
        reader=file.readlines()
    for r in reader:
        r_list=r.split("::")
        x.append([int(r_list[0]),int(r_list[1]),float(r_list[2])])
    data = np.array(x, dtype=float)
    max_list = list(np.max(data, axis=0))
    n = max_list[0]
    m = max_list[1]
    V = np.zeros((int(n), int(m)))
    for row in x:
        V[int(row[0]) - 1][int(row[1]) - 1] = float(row[2])

    V_new = NMF(V)
    old_list = list(V[query_num])
    new_list = list(V_new[query_num])
    max = 0
    index = 0
    for i in range(len(old_list)):
        if old_list[i] == 0 and new_list[i] > max:
            max = new_list[i]
            index = i
    print("为用户" + str(query_num) + '推荐电影' + str(index) + ',预测评分为' + str(max))

    # print(list(V[query_num]))
    # print(list(V_new[query_num]))

    # for row in V_new:
    #     l = list(row)
    #     print(max(l), l.index(max(l)))
    # with open('text.txt', 'w', encoding='utf-8') as fp:
    #     for row in V_new:
    #         fp.write(str(list(row)) + '\n')

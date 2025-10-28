import csv
import random
import math
import numpy as np
import pandas as pd


def NMF():
    K = 10
    times = 100
    alpha = 0.005
    beta = 0.05
    e=1e-7
    with open('ratings.csv', 'r') as file:
        reader = csv.reader(file)
        x = list(reader)
    data = np.array(x[1:], dtype=float)
    max_list = list(np.max(data, axis=0))
    n = max_list[0]
    m = max_list[1]
    V = np.zeros((int(n), int(m)))
    for row in x[1:]:
        V[int(row[0]) - 1][int(row[1]) - 1] = float(row[2])
    W = np.random.rand(int(n), K)
    H = np.random.rand(int(m), K).T
    for t in range(times):
        vht=np.dot(V,H.T)
        whht=np.dot(W,np.dot(H,H.T))
        wtv=np.dot(W.T,V)
        wtwh=np.dot(W.T,np.dot(W,H))
        for i in range(int(n)):
            for k in range(K):
                g=W[i][k]*(vht[i][k]/(e+whht[i][k]))
                W[i][k] = g if g>0 else 0
        for j in range(int(m)):
            for k in range(K):
                g=H[k][j]*(wtv[k][j]/(e+wtwh[k][j]))
                H[k][j] = g if g>0 else 0
        V_new = np.dot(W, H)
        rmse=0
        mae=0
        cnt=0
        for i in range(int(n)):
            for j in range(int(m)):
                if V[i][j]>0:
                    rmse+=pow((V[i][j]-V_new[i][j]),2)
                    mae+=abs(V[i][j]-V_new[i][j])
                    cnt+=1
        rmse=np.sqrt(rmse/cnt)
        mae=mae/cnt
        print('times=' + str(t) + ' rmse=' + str(rmse)+' mae='+str(mae))


if __name__ == '__main__':
    NMF()

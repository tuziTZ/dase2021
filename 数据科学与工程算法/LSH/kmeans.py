import random
import numpy as np


def jcd(vec1, vec2):
    intersection = 0
    union = 0
    for i in range(0, len(vec1)):
        if (vec1[i] == 1) & (vec2[i] == 1):
            intersection += 1
            union += 1
        elif (vec1[i] == 1) | (vec2[i] == 1):
            union += 1
    return intersection / (union * 1.0)


def calcDis(dataSet, centroids, k):
    clalist = []
    for data in dataSet:
        distance = []
        for i in range(k):
            distance.append(jcd(data, centroids[i]))
        clalist.append(distance)
    clalist = np.array(clalist)  # 返回一个每个点到质点的距离len(dateSet)*k的数组

    return clalist


# 计算质心
def classify(dataSet, centroids, k):
    # 计算样本到质心的距离
    # 分组并计算新的质心
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enumerate()可同时遍历索引和遍历元素
        cluster[j].append(dataSet[i])
    newCentroids = []
    changed = 0
    for i in range(k):
        c = cluster[i]
        if len(c) == 0:
            while 1:
                choose = random.sample(dataSet, 1)[0]
                if choose not in newCentroids:
                    newCentroids.append(choose)  # 随机选取一个不是其他质心的点作为质心
                    break
        else:
            newCentroids.append(random.sample(c, 1)[0])
        changed += jcd(newCentroids[i], centroids[i])

    return changed, newCentroids


# 使用k-means分类
def kmeans(dataSet, k, error):
    # 随机取质心
    centroids = random.sample(dataSet, k)

    # 更新质心 直到变化量全为0
    changed, newCentroids = classify(dataSet, centroids, k)
    # while np.any(changed > error):
    #     changed, newCentroids = classify(dataSet, newCentroids, k)

    centroids = sorted(newCentroids)  # tolist()将矩阵转换成列表 sorted()排序

    # 根据质心计算每个集群
    cluster = []
    clalist = calcDis(dataSet, centroids, k)  # 调用欧拉距离
    minDistIndices = np.argmin(clalist, axis=1)
    for i in range(k):
        cluster.append([])
    for i, j in enumerate(minDistIndices):  # enymerate()可同时遍历索引和遍历元素
        cluster[j].append({i: dataSet[i]})

    return centroids, cluster

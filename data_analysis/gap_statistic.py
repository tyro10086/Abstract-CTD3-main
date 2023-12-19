import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

from data_analysis import utils_data


#   计算簇内距离的方差
def pair_wise_distance(cluster):
    avg = cluster.mean(axis=0)
    Dk = sum(sum((cluster - avg) ** 2))
    return Dk

#   按照聚类结果分簇，分簇集计算距离方差，返回总方差
def compute_tot_d(data, classfication_result):
    res = 0
    label_set = set(classfication_result)
    for label in label_set:
        each_cluster = data[classfication_result == label, :]
        res = res + pair_wise_distance(each_cluster)
    return res

# 找到合适的中心数就停止，节约时间
def compute_gap(states, min_clusters, max_clusters, B=10):
    """
    states：聚类数据
    B：生成随机数据次数
    min(max)_clusters：聚类中心数目的区间
    """
    #   所有尝试的聚类中心数
    K = range(min_clusters, max_clusters + 1)
    #   gap值、实际数据的距离方差、随机数据方差的标准差
    gaps = np.zeros(len(K))
    Dk = np.zeros(len(K))
    sk = np.zeros(len(K))

    #   生成随机数据
    shape = states.shape
    tops = states.max(axis=0)
    bots = states.min(axis=0)
    dists = np.matrix(np.diag(tops - bots))
    rands = np.random.random_sample(size=(B, shape[0], shape[1]))
    for i in range(B):
        rands[i, :, :] = rands[i, :, :] * dists + bots

    for i, k in enumerate(K):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(states)
        label = kmeans.labels_
        # 将所有簇内的Dk存储起来
        Dk[i] = compute_tot_d(states, label)

        #   随机数据的聚类方差
        rand_Dk = np.zeros(B)
        #   生成B次随机数据，记录rand_Dk
        for j in range(B):
            rand_data = rands[j, :, :]
            kmeans.fit(rand_data)
            rand_label = kmeans.labels_
            rand_Dk[j] = compute_tot_d(rand_data, rand_label)

        #   按公式计算gaps和sk
        gaps[i] = np.log(rand_Dk).mean() - np.log(Dk[i])
        sk[i] = np.std(np.log(rand_Dk)) * np.sqrt(1 + 1.0 / B)

        #   满足条件就返回
        if i != 0 and gaps[i - 1] >= gaps[i] - sk[i]:
            return k

    print("设置的最大中心数还不够大")
    return max_clusters

#   所有可能的中心数都试一遍，可以画图
def gap_statistic(states, min_clusters, max_clusters, B=10):
    #   所有尝试的聚类中心数
    K = range(min_clusters, max_clusters + 1)
    #   gap值、实际数据的距离方差、随机数据方差的标准差
    gaps = np.zeros(len(K))
    Dk = np.zeros(len(K))
    rand_Dk = np.zeros((len(K), B))
    sk = np.zeros(len(K))

    # 生成随机数据
    shape = states.shape
    tops = states.max(axis=0)
    bots = states.min(axis=0)
    dists = np.matrix(np.diag(tops - bots))
    rands = np.random.random_sample(size=(B, shape[0], shape[1]))
    for i in range(B):
        rands[i, :, :] = rands[i, :, :] * dists + bots

    for i, k in enumerate(K):
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(states)
        label = kmeans.labels_
        # 将所有簇内的Dk存储起来
        Dk[i] = compute_tot_d(states, label)

        # 通过循环，计算每一个参照数据集下的各簇Wk值
        for j in range(B):
            rand_data = rands[j, :, :]
            kmeans.fit(rand_data)
            rand_label = kmeans.labels_
            rand_Dk[i, j] = compute_tot_d(rand_data, rand_label)

    # 计算gaps、sd_ks、sk和gapDiff
    gaps = (np.log(rand_Dk)).mean(axis=1) - np.log(Dk)
    sd_ks = np.std(np.log(rand_Dk), axis=1)
    sk = sd_ks * np.sqrt(1 + 1.0 / B)

    # 用于判别最佳k的标准，当gapDiff首次为正时，对应的k即为目标值
    gapDiff = gaps[:-1] - gaps[1:] + sk[1:]

    #   绘制gapDiff的折线图
    x = K[1:]
    plt.plot(x, gapDiff)
    plt.xlabel('簇的个数')
    plt.ylabel('K的选取')
    plt.show()

    #   找到第一个大于等于0的下标
    ind = np.argmax(gapDiff >= 0)

    #   gapDiff都小于0的情况
    if ind == 0 and gapDiff[0] < 0:
        print("设置的最大中心数还不够大")
        return max_clusters

    return x[ind]



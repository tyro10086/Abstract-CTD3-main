import csv

import numpy as np

from data_analysis import utils_data
import joblib
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from data_analysis import utils_data
# 自定义距离度量函数
from sklearn.metrics import pairwise_distances


def get_acc_states(csv_path, frequency, rel_dis, rel_speed, next_rel_dis, next_rel_speed, cost):
    state = utils_data.get_csv_info(csv_path, frequency, rel_dis, rel_speed)

    next_state = utils_data.get_csv_info(csv_path, frequency, next_rel_dis, next_rel_speed, cost)
    for item in state:
        item.append(0)

    for item in next_state:
        if item[-1] == 100:
            item[-1] = 1

    state.extend(next_state)

    state = utils_data.eliminate_similar(state, 0.001, 0.001, 1)

    return state


def custom_distance(x, y):
    # 计算自定义距离度量，这里以示例的曼哈顿距离为例
    return np.abs(x - y).sum()


def k_means_abstract(states, num_clusters):
    data = np.array(states)

    # 定义要聚类的簇数
    # num_clusters = 19727
    # 计算自定义距离矩阵
    distances = pairwise_distances(data, metric=custom_distance)
    # 创建 K-means 模型并进行训练
    kmeans = KMeans(n_clusters=num_clusters, init='random', algorithm='auto')
    kmeans.fit(data)
    # 保存模型
    joblib.dump(kmeans, 'kmeans_model.pkl')

    # 预测数据点的簇标签
    labels = kmeans.labels_

    # 获取聚类中心点的坐标
    centroids = kmeans.cluster_centers_

    # 绘制数据点和聚类中心
    plt.scatter(data[:, 0], data[:, 1], c=labels)
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', color='red')
    plt.show()


def get_raw_distribution(states):
    # 原始数据
    data = np.array(states)

    # 提取数据的 x 和 y 坐标以及颜色标签
    x = data[:, 0]
    y = data[:, 1]
    colors = data[:, 2]

    # 根据颜色标签绘制数据点
    plt.scatter(x, y, c=colors)
    plt.xlabel('rel_distance')
    plt.ylabel('rel_speed')
    plt.title('Data Distribution with Different Colors')
    plt.show()

if __name__ == '__main__':
    states = get_acc_states('td3_risk_acc_logs.csv', 5, 'rel_dis', 'rel_speed', 'next_rel_dis',
                            'next_rel_speed', 'cost')
    filename = 'MDP_of_sampled_acc.csv'
    # 使用CSV模块写入CSV文件
    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(states)
    # for state in states:
    #     if state[-1] != 0:
    #         print(state)
    # k_means_abstract(states, 20)
    # get_raw_distribution(states)

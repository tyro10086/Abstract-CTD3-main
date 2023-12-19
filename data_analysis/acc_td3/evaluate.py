import copy
import os
from argparse import ArgumentParser

import gym

import highway_env
import joblib
import matplotlib.pyplot as plt
import numpy as np
import re
import csv
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import utils
from algo import *
from data_analysis import utils_data, canopy_cluster
from data_analysis.acc_td3.mdp.construct_mdp import generate_graph_from_csv
from data_analysis.gap_statistic import compute_gap, gap_statistic
from eval_acc import eval_acc
from mdp.my_kmeans import CustomKMeans
from tool.tools import data_prcss

save_fig_path = "./imgs/"
save_mdl_path = "./mdls/"


def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--mode", type=str, help="the method of kmeans",
                        default="gap")

    parser.add_argument("--cluster_again", type=bool, help='cluster again or not',
                        default=True)

    parser.add_argument("--val_again", type=bool, help="generate val data again or not",
                        default=False)

    parser.add_argument("--train_again", type=bool, help="generate train data again or not",
                        default=False)

    parser.add_argument("--checkpoint", type=str, help="the path to save project",
                        default="project/20220931-TD3_risk-acc/checkpoint")

    parser.add_argument("--model_config", type=str, help="path of rl algorithm configuration",
                        default="conf/algorithm/TD3_risk_acc.yaml")

    parser.add_argument("--eval_config", type=str, help="path of paramaters of intervalize algorithm configuration",
                        default="conf/eval/highway_acc_eval.yaml")

    parser.add_argument("--env_config", type=str, help="path of highway env",
                        default="conf/env/highway_acc_continuous_acceleration.yaml")
    parser.add_argument("--gpu", type=str, help="[0,1,2,3 | -1] the id of gpu to train/test, -1 means using cpu",
                        default="-1")

    args = parser.parse_args()
    return args


def compute_mae(y_true, y_predict, config):
    if not config.algo:
        if config.mode == 'gap':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_gap.pkl')
        elif config.mode == 'canopy':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_canopy.pkl')
        elif config.mode == 'elbow':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_elbow.pkl')
        elif config.mode == 'silhouette':
            mdl = joblib.load(save_mdl_path + 'trad_kmeans_silhouette.pkl')
        else:
            print("ERROR: check the value of parameter mode")
            exit(0)
    # elif config.algo:
    #     if config.mode == 'gap':
    #         mdl = joblib.load(save_mdl_path + 'trad_kmeans_gap.pkl')
    #     elif config.mode == 'canopy':
    #         mdl = joblib.load(save_mdl_path + 'trad_kmeans_canopy.pkl')
    #     elif config.mode == 'elbow':
    #         mdl = joblib.load(save_mdl_path + 'trad_kmeans_elbow.pkl')
    #     elif config.mode == 'silhouette':
    #         mdl = joblib.load(save_mdl_path + 'trad_kmeans_silhouette.pkl')
    #     else:
    #         print("ERROR: check the value of parameter mode")
    #         exit(0)

    mae = 0

    cluster_centers = mdl.cluster_centers_
    mae += sum(sum(abs(y_predict - y_true)))
    ret = mae / len(cluster_centers)
    print("the value of MAE is: %s" % ret)


def plt_canopy(states, graph, config):
    path = os.path.join("./../../", config.eval_config)
    eval_config = utils.load_yml(path)

    #   先canopy算法粗聚类
    t1 = 0.4
    t2 = 0.2
    gc = canopy_cluster.Canopy(states)
    gc.setThreshold(t1, t2)
    #   canopies是列表，里面元素元组形式，分别表示中心和簇
    canopies = gc.clustering()
    print('Get %s initial centers.' % len(canopies))

    #   提取中心点，kmeans聚类
    center_point = [np.array(clu[0]) for clu in canopies]

    kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=len(canopies), initial_centers=center_point)
    kmeans.process()

    #   保存模型
    joblib.dump(kmeans, save_mdl_path + 'test_kmeans_canopy.pkl')

    # 获取中心点
    centroids = kmeans.get_centers()
    centroids = np.array(kmeans.revised_centers(centroids))

    # 获取标签
    clusters = kmeans.get_clusters()
    num_clsuters = len(clusters)
    labels = [0 for _ in range(states.shape[0])]
    for k, item in enumerate(clusters):
        for it in item:
            labels[it] = k

    #   画图
    plt.scatter((states[:, 0] + states[:, 1]) / 2, (states[:, 2] + states[:, 3]) / 2, c=labels)
    plt.scatter((centroids[:, 0] + centroids[:, 1]) / 2, (centroids[:, 2] + centroids[:, 3]) / 2, marker='X', color='red')

    plt.xlabel('rel_distance')
    plt.ylabel('rel_speed')
    plt.title('canopy')
    plt.savefig(save_fig_path + "test_result-canopy-%s" % num_clsuters)

    plt.show()

    print('The number of cluster centers is %s.' % num_clsuters)


def plt_elbow(states, graph, config):
    path = os.path.join("./../../", config.eval_config)
    eval_config = utils.load_yml(path)

    min_clusters = 2
    max_clusters = 5
    #   记录簇间距离平方和
    inertias = []

    for i in range(min_clusters, max_clusters + 1):
        kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=i)
        kmeans.process()
        # 获取中心点
        centroids = kmeans.get_centers()
        centroids = np.array(kmeans.revised_centers(centroids))

        # 获取标签
        clusters = kmeans.get_clusters()
        labels = [0 for _ in range(states.shape[0])]
        for k, item in enumerate(clusters):
            for it in item:
                labels[it] = k

        inertia = kmeans.compute_inertia(states)
        inertias.append(inertia)

    # 绘制肘方法图像
    x = range(min_clusters, max_clusters + 1)
    plt.plot(x, inertias, marker='o')
    #   标注坐标
    for i in range(len(x)):
        plt.text(x[i], inertias[i], f'({x[i]}, {int(inertias[i])})', fontsize=8, ha='center', va='top')

    plt.xlabel('Number of Clusters')
    plt.ylabel('Intra-cluster Sum of Squares')
    plt.title('Selecting n_clusters with the Elbow Method')
    plt.show()

    #   根据图像选择中心数量
    print("According to the graph, the best n_clusters is:")
    str = input()

    #   匹配输入是否是正整数
    pattern = "^[1-9]\d*$"
    mtch = re.match(pattern, str)

    #   若输入合法，确定了中心数量，进行聚类
    if mtch and int(str) <= max_clusters:
        best_n_clusters = int(str)
        best_kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=best_n_clusters)
        best_kmeans.process()

        #   保存模型
        joblib.dump(best_kmeans, save_mdl_path + 'test_kmeans_elbow.pkl')

        # 获取中心点
        centroids = best_kmeans.get_centers()
        centroids = np.array(best_kmeans.revised_centers(centroids))

        # 获取标签
        clusters = best_kmeans.get_clusters()
        num_clsuters = len(clusters)
        labels = [0 for _ in range(states.shape[0])]
        for k, item in enumerate(clusters):
            for it in item:
                labels[it] = k

        #   画图
        plt.scatter((states[:, 0] + states[:, 1]) / 2, (states[:, 2] + states[:, 3]) / 2, c=labels)
        plt.scatter((centroids[:, 0] + centroids[:, 1]) / 2, (centroids[:, 2] + centroids[:, 3]) / 2, marker='X',
                    color='red')

        plt.xlabel('rel_distance')
        plt.ylabel('rel_speed')
        plt.title('elbow')
        plt.savefig(save_fig_path + "test_result-elbow-%s" % num_clsuters)

        plt.show()

        print('The number of cluster centers is %s.' % num_clsuters)

    else:
        print("Input error")
        exit(0)


def plt_Silhouette(states, graph, config):
    path = os.path.join("./../../", config.eval_config)
    eval_config = utils.load_yml(path)

    min_clusters = 2
    max_clusters = 10
    # 初始化列表来保存每个聚类个数对应的轮廓系数
    silhouette_scores = []

    # 计算每个聚类个数对应的轮廓系数
    for n_clusters in range(min_clusters, max_clusters + 1):
        kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=n_clusters)
        kmeans.process()

        #   各个点相互之间距离
        distance_array = kmeans.pairwise_distance(states)

        #   获取标签
        clusters = kmeans.get_clusters()
        labels = [0 for _ in range(states.shape[0])]
        for k, item in enumerate(clusters):
            for it in item:
                labels[it] = k

        silhouette_avg = silhouette_score(distance_array, labels, metric="precomputed")
        silhouette_scores.append(silhouette_avg)

    #   轮廓系数画图
    plt.plot(range(min_clusters, max_clusters + 1), silhouette_scores, marker='o')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Method')
    plt.show()

    # 分数最大的是最佳聚类个数
    best_n_clusters = np.argmax(silhouette_scores) + min_clusters

    # 使用最佳聚类个数重新训练模型
    best_kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=best_n_clusters)
    best_kmeans.process()

    #   保存模型
    joblib.dump(best_kmeans, save_mdl_path + 'test_trad_kmeans_silhouette.pkl')

    # 获取中心点
    centroids = best_kmeans.get_centers()
    centroids = np.array(best_kmeans.revised_centers(centroids))

    # 获取标签
    clusters = best_kmeans.get_clusters()
    num_clsuters = len(clusters)
    labels = [0 for _ in range(states.shape[0])]
    for k, item in enumerate(clusters):
        for it in item:
            labels[it] = k

    #   画图
    plt.scatter((states[:, 0] + states[:, 1]) / 2, (states[:, 2] + states[:, 3]) / 2, c=labels)
    plt.scatter((centroids[:, 0] + centroids[:, 1]) / 2, (centroids[:, 2] + centroids[:, 3]) / 2, marker='X',
                color='red')

    plt.xlabel('rel_distance')
    plt.ylabel('rel_speed')
    plt.title('silhouette')
    plt.savefig(save_fig_path + "test_result-silhouette-%s" % num_clsuters)

    plt.show()

    print('The number of cluster centers is %s.' % num_clsuters)


def plt_gap(states, graph, config):
    path = os.path.join("./../../", config.eval_config)
    eval_config = utils.load_yml(path)

    min_clusters = 2
    max_clusters = 18
    gap_states = np.zeros((states.shape[0], 2))
    gap_states[:, 0] = (states[:, 0] + states[:, 1]) / 2
    gap_states[:, 1] = (states[:, 2] + states[:, 3]) / 2
    num_cluster = compute_gap(gap_states, min_clusters, max_clusters)
    print(num_cluster)
    kmeans = CustomKMeans(config=eval_config, data=states, graph=graph, k=num_cluster)
    kmeans.process()

    #   保存模型
    joblib.dump(kmeans, save_mdl_path + 'test_kmeans_gap.pkl')

    # 获取中心点
    centroids = kmeans.get_centers()
    centroids = np.array(kmeans.revised_centers(centroids))

    # 获取标签
    clusters = kmeans.get_clusters()
    num_clsuters = len(clusters)
    labels = [0 for _ in range(states.shape[0])]
    for k, item in enumerate(clusters):
        for it in item:
            labels[it] = k

    #   画图
    plt.scatter((states[:, 0] + states[:, 1]) / 2, (states[:, 2] + states[:, 3]) / 2, c=labels)
    plt.scatter((centroids[:, 0] + centroids[:, 1]) / 2, (centroids[:, 2] + centroids[:, 3]) / 2, marker='X',
                color='red')

    plt.xlabel('rel_distance')
    plt.ylabel('rel_speed')
    plt.title('gap statistic')
    plt.savefig(save_fig_path + "test_result-gap-%s" % num_clsuters)

    plt.show()

    print('The number of cluster centers is %s.' % num_clsuters)


def draw(states, graph, config):
    if config.mode == 'gap':
        plt_gap(states, graph, config)
    elif config.mode == 'elbow':
        plt_elbow(states, graph, config)
    elif config.mode == 'canopy':
        plt_canopy(states, graph, config)
    elif config.mode == 'silhouette':
        plt_Silhouette(states, graph, config)
    else:
        print("ERROR: check the value of parameter mode")
        exit(0)


if __name__ == '__main__':
    # data = get_acc_states('td3_risk_acc_logs.csv', 1, 'rel_dis', 'rel_speed', 'next_rel_dis',
    #                         'next_rel_speed', 'cost')
    # filename = 'MDP_of_sampled_acc.csv'
    # data = np.array(data)
    # print(data.shape[0])
    # # 使用CSV模块写入CSV文件
    # with open(filename, 'w', newline='') as file:
    #     writer = csv.writer(file)
    #     writer.writerows(data)

    config = parse_args()

    log_file_train = '../../my_log_ACC.log'
    csv_file_train = './td3_risk_acc_logs.csv'
    csv_file_phase = './result.csv'
    # csv_file_phase = './first_phase_result.csv'

    #   训练好的模型里面的
    log_file_val = '../../trained_acc_data.log'
    csv_file_val = './td3_risk_acc_val_logs.csv'

    utils_data.log2csv(log_file_train, csv_file_train)

    # 如果重新训练了数据，就要把这些raw data经过第一阶段处理
    if config.train_again:
        path = os.path.join("./../../", config.eval_config)
        eval_config = utils.load_yml(path)


        utils_data.log2csv(log_file_train, csv_file_train)
        prcss = data_prcss(eval_config["dim"]["total_dim"], eval_config["dim"]["state_dim"],eval_config["dim"]["action_dim"],
                           eval_config["dim"]["reward_dim"], eval_config["dim"]["cost_dim"],
                           eval_config["granularity"]["state_gran"], eval_config["granularity"]["action_gran"],
                           eval_config["granularity"]["reward_gran"], eval_config["granularity"]["cost_gran"],
                           eval_config["upperbound"]["state_upperbound"], eval_config["lowerbound"]["state_lowerbound"],
                           eval_config["upperbound"]["action_upperbound"], eval_config["lowerbound"]["action_lowerbound"],
                           eval_config["upperbound"]["reward_upperbound"], eval_config["lowerbound"]["reward_lowerbound"],
                           eval_config["upperbound"]["cost_upperbound"], eval_config["lowerbound"]["cost_lowerbound"],)

        #TODO: 太他妈丑了 可以改成在init中解析

        prcss.read_in(csv_file_train)
        prcss.process()
        prcss.write_csv(csv_file_phase)

    #   构建MDP模型
    graph = generate_graph_from_csv(csv_file_phase)

    #   生成聚类用数据
    train_data = []
    for tup in graph.keys():
        train_data.append([ele for inner_tuple in tup for ele in inner_tuple])
    train_data = np.array(train_data)

    if config.cluster_again:
        draw(train_data, graph, config)

    if config.val_again:
        eval_acc(config)

    utils_data.log2csv(log_file_val, csv_file_val)
    val_data = utils_data.get_csv_info(csv_file_val, 1, 'rel_dis_true', 'rel_speed_true',
                                       'rel_dis_predict', 'rel_speed_predict')
    val_data = np.array(val_data)

    y_true = val_data[:, 0:2]
    y_predict = val_data[:, 2:4]

    compute_mae(y_true, y_predict, config)
    # TODO：查找节点时多线程，
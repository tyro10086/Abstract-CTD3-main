import copy
import math
import os

import numpy as np
import pickle

from pyclustering.cluster.center_initializer import kmeans_plusplus_initializer
from pyclustering.cluster.kmeans import kmeans, kmeans_visualizer
from pyclustering.cluster import cluster_visualizer
from pyclustering.utils import read_sample
from pyclustering.utils.metric import distance_metric, type_metric
from pyclustering.samples.definitions import SIMPLE_SAMPLES

from scipy.spatial.distance import jensenshannon

import sys

import utils
from data_analysis.acc_td3.mdp.construct_mdp import generate_graph_from_csv


#   sys.path.append("F:\桌面\Abstract-CTD3-main-master\data_analysis\\acc_td3")


# 用于计算交集
class MySet:
    # 根据元组创建集合
    def __init__(self, edge):
        self.mins = edge.action[0][0]
        self.maxs = edge.action[0][1]
        self.reward = edge.reward

    def getMin(self):
        return self.mins

    def getMax(self):
        return self.maxs

    def getReward(self):
        return self.reward


def getIntersection(l1, l2):  # 求交集方法,同时计算奖励的最大差值->由于粒度一致
    result = []  # 用来存储l1和l2的所有交集

    # 对输入的两个列表进行排序
    l1.sort(key=lambda x: x.getMin())
    l2.sort(key=lambda x: x.getMin())

    i = 0
    j = 0
    while i < len(l1) and j < len(l2):
        s1 = l1[i]  # 在方法里调用MySet类
        s2 = l2[j]
        if s1.getMin() < s2.getMin():
            if s1.getMax() < s2.getMin():  # 第一种时刻，交集为空，不返回
                i += 1
            elif s1.getMax() <= s2.getMax():  # 第二种时刻
                result.append(MySet(l2[j]))
                i += 1
            else:  # 第三种时刻第二种情况
                result.append(MySet(l2[j]))
                j += 1
        elif s1.getMin() <= s2.getMax():
            if s1.getMax() <= s2.getMax():  # 第三种时刻第一种情况
                result.append(MySet(l1[j]))
                i += 1
            else:  # 第四种时刻
                result.append(MySet(l1[j]))
                j += 1
        else:  # 第五种时刻
            j += 1
    return result


#   把图还原成2darray
def graph2arr(graph):
    lists = []
    for key, value in graph.items():
        ls = []
        #   处理state
        for it in value.state:
            ls = ls + [*it]

        for edge in value.edges:
            #   处理next_state
            ls1 = copy.deepcopy(ls)
            for it in edge.next_node.state:
                ls1 = ls1 + [*it]
            #   处理action reward cost done prob
            for i in range(len(edge.action)):
                ls2 = copy.deepcopy(ls1)
                ls2 = ls2 + [*edge.action[i]]
                ls2 = ls2 + [*edge.reward[i]]
                ls2 = ls2 + [*edge.cost[i]]
                ls2.append(1 if edge.done is True else 0)
                ls2.append(edge.prob)

                lists.append(ls2)

    ret = np.array(lists)
    return ret


def manhattanDistance(vec1, vec2):
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    return abs((vec1 - vec2)).sum()



class CustomKMeans(kmeans):
    def __init__(self, config, graph, data, k=8, initial_centers=None, tolerance=0.001, ccore=True):
        # 初始化中心点和距离函数
        if not initial_centers:
            initial_centers = kmeans_plusplus_initializer(data, k).initialize()
        metric = distance_metric(type_metric.USER_DEFINED, func=self.distance)

        super().__init__(data, initial_centers, tolerance, ccore, metric=metric)
        self.cr = 0.5
        self.cd = 0.5
        self.cp = 0.5
        self.cs = 0.5
        self.config = config
        self.state_dim = config["dim"]["state_dim"]
        self.graph = graph
        self.states = set()
        for tup in graph.keys():
            self.states.add(tup)
        self.precs = self.get_prec(config)

    #   获取保留几位小数的精度
    def get_prec(self, config):
        ret = []
        gran = config["granularity"]["state_gran"]
        for i in range(self.state_dim):
            ret.append(math.ceil(math.log10(gran[i]) * -1))
        return ret

    #   为输入匹配mdp状态最相近的节点
    def matchnode(self, data):
        """
        data 待匹配状态 1darray
        return 匹配到的节点 node
        """
        data_tup = (tuple(data[0:self.state_dim]), tuple(data[self.state_dim:]))
        if data_tup in self.graph.keys():
            return self.graph[data_tup]
        # 从这里开始多线程
        # self.cnt += 1
        # if self.cnt % 100 == 0:
        #     print(self.cnt)

        min_state = None
        min_distance = 100
        for k, item in enumerate(self.states):
            it = [ele for inner_tuple in item for ele in inner_tuple]

            dis = manhattanDistance(it, data)

            if dis < min_distance:
                min_state = item
                min_distance = dis
        # 多线程结束 发送到主线程，开始比较
        return self.graph[min_state]

    #   计算mdp中两个节点距离
    def distance(self, data1, data2):
        # 判断两个状态是否相同
        x = self.matchnode(data1)
        y = self.matchnode(data2)
        if x.state == y.state:
            return 0

        lx, ly = [], []  # 统计动作区间
        prob_x, prob_y = [], []  # 统计概率
        action_x, action_y = [], []  # action具体值
        max_reward_difference = 0  # 奖励的最大差值

        # 读取动作元组,概率分布
        for edge_x in x.edges:
            lx.append(MySet(edge_x))
            prob_x.append(edge_x.prob)
            action_x.append((edge_x.action[0][0] + edge_x.action[0][1]) / 2)
        for edge_y in y.edges:
            lx.append(MySet(edge_y))
            prob_y.append(edge_y.prob)
            action_y.append((edge_y.action[0][0] + edge_y.action[0][1]) / 2)
        result = getIntersection(lx, ly)  # 动作区间交集

        # 需要保证动作有交集，才有最大奖励差
        if len(result) != 0:
            # 计算奖励的最大差值
            min_reward = result[0].getReward()
            max_reward = result[0].getReward()
            for customSet in result:
                if min_reward > customSet.getReward():
                    min_reward = customSet.getReward()
                if max_reward < customSet.getReward():
                    max_reward = customSet.getReward()
            max_reward_difference = max_reward - min_reward

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x.state) - np.array(y.state))

        if not x.edges or not y.edges:
            return self.cr * state_distance

        # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
        max_len = max(len(prob_x), len(prob_y))

        #   保证长度一样，不一样补0到一样
        if len(prob_x) == max_len:
            for _ in range(max_len - len(prob_y)):
                prob_y.append(0)
        else:
            for _ in range(max_len - len(prob_x)):
                prob_x.append(0)

        distribution_difference = jensenshannon(prob_x, prob_y)

        max_action_difference = max(abs(max(action_x) - min(action_y)), abs(min(action_x) - max(action_y)))
        #    print(self.cr * state_distance + max_reward_difference + self.cd * max_action_difference + self.cd * distribution_difference)
        return self.cs * state_distance + self.cr * max_reward_difference + self.cd * max_action_difference + self.cp * distribution_difference

    # 将聚类得到的中心点变成mdp模型中结点
    def revised_centers(self, center):
        """
        centers/centroids: list[list[float]]
        """
        centroids = []
        for item in center:
            rounded_item = []
            for i in range(self.state_dim):
                rounded_item.append(round(item[2 * i], self.precs[i]))
                rounded_item.append(round(item[2 * i + 1], self.precs[i]))
            node = self.matchnode(rounded_item)
            cluster = [ele for inner_tuple in node.state for ele in inner_tuple]
            centroids.append(cluster)
        return centroids


    #   计算簇间距离，用自定义距离函数
    def compute_inertia(self, data):
        """
        data: 原始数据 2darray
        """
        # 获取中心点
        inertia = 0
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))

        # 获取标签
        clusters = self.get_clusters()
        labels = [0 for _ in range(data.shape[0])]
        for k, item in enumerate(clusters):
            for it in item:
                labels[it] = k

        for k, item in enumerate(labels):
            inertia += self.distance(data[k], centroids[item])

        return inertia

    #   在聚类之后使用，返回稠密的距离矩阵
    def tranform(self, data):
        """
        data 输入数据 2darray
        ret 距离矩阵 2darray (n_sample, n_center) 表示每个输入数据到中心点距离
        """
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))

        distances = np.zeros((data.shape[0], centroids.shape[0]))
        for i in range(centroids.shape[0]):
            distances[:, i] = np.array([self.distance(item, centroids[i]) for item in data])
        return distances

    #   用自定义距离公式计算输入各个点的距离
    def pairwise_distance(self, data):
        """
        data 输入数据 2darray
        ret 距离矩阵 有对称性质 2darray (n_sample, n_sample) 每个输入数据到其它输入数据距离
        """
        n = data.shape[0]
        ret = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i, n):
                dis = self.distance(data[i], data[j])
                ret[i][j] = dis
                ret[j][i] = dis
        return ret



if __name__ == '__main__':
    graph = generate_graph_from_csv("./../result.csv")
    data = []
    for tup in graph.keys():
        data.append([ele for inner_tuple in tup for ele in inner_tuple])
    data = np.array(data)

    path = os.path.join("./../../../", "conf/eval/highway_acc_eval.yaml")
    eval_config = utils.load_yml(path)

    # print(graph_data)
    kmeans_instance = CustomKMeans(config=eval_config, data=data, graph=graph, k=3)

    # run cluster analysis and obtain results
    kmeans_instance.process()
    centers = kmeans_instance.get_centers()
    print(centers)
    print(kmeans_instance.revised_centers(centers))
    centers = np.array(centers)
    print(centers.shape)


    print("成功！！")

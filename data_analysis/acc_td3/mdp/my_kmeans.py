import copy
import math
import os

import joblib
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
from data_analysis.acc_td3.mdp.construct_mdp import generate_graph_from_csv, generate_states_from_graph, reshape_graph


#   sys.path.append("F:\桌面\Abstract-CTD3-main-master\data_analysis\\acc_td3")


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
    def __init__(self, config, graph, datas, k=8, initial_centers=None, tolerance=0.001, ccore=True):
        """
        # config 一阶段抽象的参数conf/eval/...yaml
        # graph dict一阶段抽象后生成的mdp图 key:((upperbound, lowerbound), ...) value:node
        # data 2d-array 把graph.keys()里面的tuple变成list 然后2d列表转数组
        """
        # 初始化中心点和距离函数
        if not initial_centers:
            initial_centers = kmeans_plusplus_initializer(datas, k).initialize()
        metric = distance_metric(type_metric.USER_DEFINED, func=self.distance)

        super().__init__(datas, initial_centers, tolerance, ccore, metric=metric)
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
        """
        输入config是一阶段抽象里面的参数，用里面的粒度来计算保留精度
        """
        ret = []
        gran = config["granularity"]["state_gran"]
        for i in range(self.state_dim):
            ret.append(math.ceil(math.log10(gran[i]) * -1))
        return ret

    #   为输入匹配mdp状态最相近的节点
    def matchnode(self, data):
        """
        data 待匹配状态 1darray 从init的data里面提取出来的一行
        return 匹配到的节点 node
        """
        #   数据改成活的
        data_list = []
        for i in range(0, self.state_dim*2, 2):
            data_list.append((data[i], data[i+1]))
        data_tup = tuple(data_list)

        if data_tup in self.graph.keys():
            return self.graph[data_tup]
        # 从这里开始多线程
        # self.cnt += 1
        # if self.cnt % 100 == 0:
        #     print(self.cnt)

        min_state = None
        min_distance = 100
        for k, item in enumerate(self.states):
            #   把元组形式的key变回tuple
            data_ls = [ele for inner_tuple in item for ele in inner_tuple]

            dis = manhattanDistance(data_ls, data)

            if dis < min_distance:
                min_state = item
                min_distance = dis
        # 多线程结束 发送到主线程，开始比较
        return self.graph[min_state]

    #   输入两个概率分布，返回差异
    def compute_distribution_difference(self, prob_x, prob_y):
        max_len = max(len(prob_x), len(prob_y))

        #   保证长度一样，不一样补0到一样
        if len(prob_x) == max_len:
            for _ in range(max_len - len(prob_y)):
                prob_y.append(0)
        else:
            for _ in range(max_len - len(prob_x)):
                prob_x.append(0)

        # 后继状态分布之间的距离——詹森-香农距离：衡量两个概率分布之间差异的距离度量，KL散度的拓展
        return jensenshannon(prob_x, prob_y)

    #   输入两个attr，找动作相交区间里面的最大奖励差
    def get_intersection_reward(self, attr_x, attr_y):
        #   不相交的情况
        if attr_x.max_action < attr_y.min_action or attr_x.min_action > attr_y.max_action:
            return 0

        #   交集的上下界
        max_action = min(attr_y.max_action, attr_x.max_action)
        min_action = max(attr_y.min_action, attr_x.min_action)

        #   初始化部分可以改进
        x_max_reward, y_max_reward = 100, 100
        x_min_reward, y_min_reward = -100, -100

        for i, action in enumerate(attr_x.actions):
            if min_action <= action <= max_action:
                x_max_reward = max(attr_x.rewards[i], x_max_reward)
                x_min_reward = min(attr_x.rewards[i], x_min_reward)

        for i, action in enumerate(attr_y.actions):
            if min_action <= action <= max_action:
                y_max_reward = max(attr_y.rewards[i], y_max_reward)
                y_min_reward = min(attr_y.rewards[i], y_min_reward)

        return max(x_max_reward - y_min_reward, y_max_reward - x_min_reward)

    #   计算mdp中两个节点距离
    def distance(self, data1, data2):
        # 判断两个状态是否相同
        x = self.matchnode(data1)
        y = self.matchnode(data2)
        if x.state == y.state:
            return 0

        # 状态本身之间的距离
        state_distance = np.linalg.norm(np.array(x.state) - np.array(y.state))

        #   如果有一个是终止节点
        if not x.actions or not y.actions:
            return self.cs * state_distance

        #   动作区间交集奖励的最大差异
        max_reward_difference = self.get_intersection_reward(x, y)  # 动作区间交集

        #   后继状态分布的差异
        distribution_difference = self.compute_distribution_difference(x.probs, y.probs)

        #   动作的最大差异
        max_action_difference = max(abs(x.max_action - y.min_action), abs(x.min_action - y.max_action))

        return self.cs * state_distance + self.cr * max_reward_difference + self.cd * max_action_difference + self.cp * distribution_difference

    # 将聚类得到的中心点变成mdp模型中结点
    def revised_centers(self, center):
        """
        centers/centroids: list[list[float]]，精度已经保留好
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
    def compute_inertia(self, datas):
        """
        datas: 原始数据 2darray 和init的输入一样
        """
        # 获取中心点
        inertia = 0
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))

        # 获取标签
        clusters = self.get_clusters()
        labels = [0 for _ in range(datas.shape[0])]
        for k, item in enumerate(clusters):
            for it in item:
                labels[it] = k

        for k, item in enumerate(labels):
            inertia += self.distance(datas[k], centroids[item])

        return inertia

    #   在聚类之后使用，返回稠密的距离矩阵，最后没用上
    def tranform(self, datas):
        """
        data 输入数据 2darray
        ret 距离矩阵 2darray (n_sample, n_center) 表示每个输入数据到中心点距离
        """
        centroids = self.get_centers()
        centroids = np.array(self.revised_centers(centroids))

        distances = np.zeros((datas.shape[0], centroids.shape[0]))
        for i in range(centroids.shape[0]):
            distances[:, i] = np.array([self.distance(item, centroids[i]) for item in datas])
        return distances

    #   用自定义距离公式计算输入各个点的距离
    def pairwise_distance(self, datas):
        """
        data 输入数据 2darray
        ret 距离矩阵 有对称性质 2darray (n_sample, n_sample) 每个输入数据到其它输入数据距离
        计算Silhouette时需要
        """
        n = datas.shape[0]
        ret = np.zeros(shape=(n, n))
        for i in range(n):
            for j in range(i, n):
                dis = self.distance(datas[i], datas[j])
                ret[i][j] = dis
                ret[j][i] = dis
        return ret



if __name__ == '__main__':
    #   获得config
    path = os.path.join("./../../../", "conf/eval/highway_acc_eval.yaml")
    eval_config = utils.load_yml(path)

    #   获得图和状态
    graph = generate_graph_from_csv("./../result.csv")
    print("图：",graph)
    #   把图修改了，依然是字典，value本来是edges，改成了一个attr类(在construct_mdp.py)
    #   attr每个状态分散的动作、奖励、转移概率整合到一起，这一步在计算距离时是无法避免的，先提前做了，省的每次重复操作
    shaped_graph = reshape_graph(graph)
    print("修正后的图:", shaped_graph)
    datas = generate_states_from_graph(graph)
    #   创建对象
    kmeans_instance = CustomKMeans(config=eval_config, datas=datas, graph=shaped_graph, k=3)

    #   进行聚类
    kmeans_instance.process()

    #   模型保存
    joblib.dump(kmeans_instance, 'test.pkl')
    #   加载模型
    mdl = joblib.load('test.pkl')

    #   获取中心点，但此时是没有修正过的
    centers = mdl.get_centers()
    print("中心：", centers)
    #   中心点修正
    revised_centers = mdl.revised_centers(centers)
    print("修正后的中心：", revised_centers)

    #   随机数据
    state = np.random.random(size=(2,))
    #   输入模型之前需要先把数据转换成一阶段抽象后的形式
    state_ = utils.intervalize_state(state, eval_config)
    print("待预测状态：", state_)

    #   预测
    #   注意predict函数的输入是list[数据点]，不是单个数据点，因此需要先套一层[]
    state_ = [state_]
    #   因此返回的label也是list[int]，需要索引0[0]
    label = mdl.predict(state_)[0]
    print("预测的簇标签：", label)
    centroid = revised_centers[label]
    print("预测的中心点：", centroid)




import math
import random
import numpy as np
import matplotlib.pyplot as plt


class Canopy:
    def __init__(self, data):
        self.data = data
        self.t1 = 0
        self.t2 = 0

    # 设置初始阈值t1 和 t2
    def setThreshold(self, t1, t2):
        if t1 > t2:
            self.t1 = t1
            self.t2 = t2
        else:
            print("t1 needs to be larger than t2!")

    # 欧式距离
    def euclideanDistance(self, vec1, vec2):
        return math.sqrt(((vec1 - vec2) ** 2).sum())

    # 根据当前dataset的长度随机选择一个下标
    def getRandIndex(self):
        return np.random.randint(len(self.data))

    # 核心算法
    def clustering(self):
        if self.t1 == 0:
            print('Please set the threshold t1 and t2!')
        else:
            canopies = []  # 用于存放最终归类的结果
            while len(self.data) != 0:
                # 获取一个随机下标
                rand_index = self.getRandIndex()
                # 随机获取一个中心点，定为P点
                current_center = self.data[rand_index]
                # 初始化P点的canopy类容器
                current_center_list = []
                # 初始化P点的删除容器
                delete_list = []
                # 删除随机选择的中心点P
                self.data = np.delete(self.data, rand_index, 0)

                for datum_j in range(len(self.data)):
                    datum = self.data[datum_j]
                    # 计算选取的中心点P到每个点之间的距离
                    distance = self.euclideanDistance(current_center, datum)
                    if distance < self.t1:
                        # 若距离小于t1，则将点归入P点的canopy类
                        current_center_list.append(datum)
                    if distance < self.t2:
                        # 若小于t2则归入删除容器
                        delete_list.append(datum_j)
                self.data = np.delete(self.data, delete_list, 0)
                canopies.append((current_center, current_center_list))

                #   删除空的簇
                canopies = [cluster for cluster in canopies if len(cluster[1]) > 1]
            return canopies



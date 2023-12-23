import csv
import ast
import re

import numpy as np


class Node:
    # def __init__(self, rel_dis, rel_speed):
    #     self.rel_dis = rel_dis
    #     self.rel_speed = rel_speed
    #     self.edges = []

    def __init__(self, *columns):
        self.state = []
        self.edges = []
        for column in columns:
            self.state.append(column)

    def add_edge(self, next_node, action, reward, done, cost, prob):
        self.edges.append(Edge(next_node, action, reward, done, cost, prob))


class Edge:
    def __init__(self, next_node, action, reward, done, cost, prob):
        self.next_node = next_node
        self.actions = action
        self.rewards = reward
        self.done = done
        self.cost = cost
        self.prob = prob

class attrs:
    def __init__(self, state, edges):
        self.state = state
        self.probs = []
        self.actions = []
        self.rewards = []
        self.max_action = -100
        self.min_action = 100
        for edge in edges:
            self.probs.append(edge.prob)
            for act in edge.actions:
                a = (act[0] + act[1]) / 2
                self.actions.append(a)
                self.min_action = min(self.min_action, a)
                self.max_action = max(self.max_action, a)
            for reward in edge.rewards:
                r = (reward[0] + reward[1]) / 2
                self.rewards.append(r)

            #   终止节点actions和rewards会为空，但是不用担心，这种情况下计算距离时只有状态差异


# 递归函数来遍历并提取 True 和 False
def extract_bool(data):
    bool_list = []
    if isinstance(data, bool):
        bool_list.append(data)
    elif isinstance(data, (list, tuple)):
        for item in data:
            bool_list.extend(extract_bool(item))
    return bool_list


def extract_attr(data):
    # 使用正则表达式提取括号内的数据
    pattern = r'\((-?\d+\.\d+), (-?\d+\.\d+)\)'
    matches = re.findall(pattern, data)
    # 将提取到的数据作为列表，列表内存储的是元组
    data_list = [tuple(float(x) for x in match) for match in matches]
    return data_list


def extract_state(data):
    # 解析字符串为 Python 对象
    tran_data = ast.literal_eval(data)
    rel_dis = tran_data[0][0]
    rel_speed = tran_data[0][1]
    next_rel_dis = tran_data[1][0]
    next_rel_speed = tran_data[1][1]
    return rel_dis, rel_speed, next_rel_dis, next_rel_speed


def extract_action_reward_cost(data, num):
    action = []
    reward = []
    cost = []
    cycle = len(data) / num
    for index, data in enumerate(data):
        if index % cycle == 0:
            action.append(data)
        if index % cycle == 1:
            reward.append(data)
        if index % cycle == 2:
            cost.append(data)
    return action, reward, cost


def generate_graph_from_csv(file_path):
    graph = {}

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取表头

        for row in reader:
            tran = row[0]
            rel_dis, rel_speed, next_rel_dis, next_rel_speed = extract_state(tran)

            attr = row[1]
            action_reward_cost = extract_attr(attr)
            # 解析字符串为 Python 对象
            attr_data = ast.literal_eval(attr)
            # 提取 True 和 False 并按出现顺序存储到列表中
            done = [value for value in extract_bool(attr_data)]

            num = int(row[2])
            action, reward, cost = extract_action_reward_cost(action_reward_cost, num)

            prob = float(row[3])

            # key表示state,改成ndarray
            current_node_key = (rel_dis, rel_speed)
            next_node_key = (next_rel_dis, next_rel_speed)
            if current_node_key not in graph:
                graph[current_node_key] = Node(rel_dis, rel_speed)

            current_node = graph[current_node_key]

            if next_node_key not in graph:
                graph[next_node_key] = Node(next_rel_dis, next_rel_speed)

            next_node = graph[next_node_key]
            current_node.add_edge(next_node, action, reward, done, cost, prob)
    return graph

#   提取graph中所有的状态 返回2darray 可以直接输入到kmeans对象中
def generate_states_from_graph(graph):
    s = set()

    for key, value in graph.items():
        s.add(key)
        s.add(tuple(value.state))
    datas = []
    for item in s:
        datas.append([ele for inner_tuple in item for ele in inner_tuple])
    return np.array(datas)

def reshape_graph(graph):
    ret = {}
    for key, value in graph.items():
        state = value.state
        edges = value.edges
        ret[key] = attrs(state, edges)

    return ret



if __name__ == '__main__':
    # graph = generate_graph_from_csv("./../first_phase_result.csv")
    graph = generate_graph_from_csv("./../result.csv")
    shaped_graph = reshape_graph(graph)
    for key, value in graph.items():
        print(key)
        print(value.state)
        for edge in value.edges:
            print("Next Node:", edge.next_node)
            print("Action:", edge.actions)
            print("Reward:", edge.rewards)
            print("Done:", edge.done)
            print("Cost:", edge.cost)
            print("Probability:", edge.prob)
    print("===========================")
    for key, value in shaped_graph.items():
        print(key)
        print(value.state)
        print(value.actions)
        print(value.rewards)
        print(value.probs)

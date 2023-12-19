import csv
import ast
import re


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
        self.action = action
        self.reward = reward
        self.done = done
        self.cost = cost
        self.prob = prob


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
    y_dis = tran_data[0][0]
    v_y = tran_data[0][1]
    cos_h = tran_data[0][2]
    next_y_dis = tran_data[1][0]
    next_v_y = tran_data[1][1]
    next_cos_h = tran_data[1][2]
    return y_dis, v_y, cos_h, next_y_dis, next_v_y, next_cos_h


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
            y_dis, v_y, cos_h, next_y_dis, next_v_y, next_cos_h = extract_state(tran)

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
            current_node_key = (y_dis, v_y, cos_h)
            next_node_key = (next_y_dis, next_v_y, next_cos_h)
            if current_node_key not in graph:
                graph[current_node_key] = Node(y_dis, v_y, cos_h)

            current_node = graph[current_node_key]

            if next_node_key not in graph:
                graph[next_node_key] = Node(next_y_dis, next_v_y, next_cos_h)

            next_node = graph[next_node_key]
            current_node.add_edge(next_node, action, reward, done, cost, prob)
    return graph


if __name__ == '__main__':

    graph = generate_graph_from_csv("./../result.csv")
    for key, value in graph.items():
        print(key)
        print(value.state)
        for edge in value.edges:
            print("Next Node:", edge.next_node)
            print("Action:", edge.action)
            print("Reward:", edge.reward)
            print("Done:", edge.done)
            print("Cost:", edge.cost)
            print("Probability:", edge.prob)

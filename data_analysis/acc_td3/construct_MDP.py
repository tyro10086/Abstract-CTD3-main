import csv
import networkx as nx
import matplotlib.pyplot as plt

class Node:
    def __init__(self, rel_dis, rel_speed):
        self.rel_dis = rel_dis
        self.rel_speed = rel_speed
        self.edges = []

    def add_edge(self, next_node, acc, reward, cost):
        self.edges.append(Edge(next_node, acc, reward, cost))

class Edge:
    def __init__(self, next_node, acc, reward, cost):
        self.next_node = next_node
        self.acc = acc
        self.reward = reward
        self.cost = cost

# 从CSV文件读取数据并生成图
def generate_graph_from_csv(file_path):
    graph = {}

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        header = next(reader)  # 读取表头

        for row in reader:
            rel_dis = float(row[0])
            rel_speed = float(row[1])
            acc = float(row[2])
            reward = float(row[3])
            cost = float(row[6])  # 修正此处为索引 6，即列号为 7 的列
            next_rel_dis = float(row[4])
            next_rel_speed = float(row[5])

            current_node_key = (rel_dis, rel_speed)
            if current_node_key not in graph:
                graph[current_node_key] = Node(rel_dis, rel_speed)
            current_node = graph[current_node_key]

            next_node_key = (next_rel_dis, next_rel_speed)
            if next_node_key not in graph:
                graph[next_node_key] = Node(next_rel_dis, next_rel_speed)
            next_node = graph[next_node_key]

            current_node.add_edge(next_node, acc, reward, cost)

    return graph

# 从CSV文件生成图数据结构
csv_file_path = '../gra_10.csv'
graph = generate_graph_from_csv(csv_file_path)

# 创建有向图对象
G = nx.DiGraph()

# 添加节点和边到图对象
for node_key, node in graph.items():
    G.add_node(node_key)
    for edge in node.edges:
        G.add_edge(node_key, (edge.next_node.rel_dis, edge.next_node.rel_speed), acc=edge.acc, reward=edge.reward, cost=edge.cost)

# 绘制图形
pos = nx.spring_layout(G)  # 选择布局算法
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', font_size=8, font_weight='bold', arrows=True)
edge_labels = {(u, v): f"加速度={data['acc']}, 奖励={data['reward']}, 成本={data['cost']}" for u, v, data in G.edges(data=True)}
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6, font_family='SimHei')

# 显示图形
plt.title('有向图')
plt.axis('off')
plt.show()


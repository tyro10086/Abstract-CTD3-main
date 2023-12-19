import numpy as np
import csv
import math
import torch
import ast


class data_prcss:
    def __init__(self, col_num, state_dim, act_dim, rewd_dim, cost_dim,
                 sub_state_range, sub_act_range, sub_rewd_range, sub_cost_range,
                 state_upbound, state_lowbound, act_upbound, act_lowbound,
                 rewd_upbound, rewd_lowbound, cost_upbound, cost_lowbound):

        self.col_num = col_num
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.rewd_dim = rewd_dim
        self.cost_dim = cost_dim


        self.sub_state_range = sub_state_range
        self.sub_act_range = sub_act_range
        self.sub_rewd_range = sub_rewd_range
        self.sub_cost_range = sub_cost_range

        self.state_upbound = state_upbound
        self.state_lowbound = state_lowbound
        self.act_upbound = act_upbound
        self.act_lowbound = act_lowbound
        self.rewd_upbound = rewd_upbound
        self.rewd_lowbound = rewd_lowbound
        self.cost_upbound = cost_upbound
        self.cost_lowbound = cost_lowbound

        self.rows = []
        self.appro_rows = []
        self.state_trans = []
        self.state_act_trans = []

        self.state_set = {}
        self.act_set = {}
        self.reward_set = {}
        self.cost_set = {}

        self.states = []
        self.acts = []
        self.rewards = []
        self.costs = []
        self.n_states = []

        self.state_dict = []
        self.act_dict = []
        self.reward_dict = []
        self.cost_dict = []

        self.states_trans_out = []

    def read_in(self, file_name):
        csv_reader = csv.reader(open(file_name))
        i = 0
        for row in csv_reader:
            if i == 0 and len(row) != self.col_num:
                print("列数对不齐")
                return
            elif i != 0:
                self.rows.append(row)
            i = i + 1

    def process(self):
        ##str2float
        for ls in self.rows:
            for i in range(self.col_num):
                if i == self.col_num - 2:
                    ls[i] = bool(ls[i])
                else:
                    ls[i] = float(ls[i])

        ##每一行分解成s a r next_s c
        for ls in self.rows:
            ind = 1
            tem = []
            for j in range(ind, ind + self.state_dim):
                tem.append(ls[j])
            self.states.append(tuple(tem))

            ind = ind + self.state_dim
            tem = []
            for j in range(ind, ind + self.act_dim):
                tem.append(ls[j])
            self.acts.append(tuple(tem))

            ind = ind + self.act_dim
            tem = []
            for j in range(ind, ind + self.rewd_dim):
                tem.append(ls[j])
            self.rewards.append(tuple(tem))

            ind = ind + self.rewd_dim
            tem = []
            for j in range(ind, ind + self.state_dim):
                tem.append(ls[j])
            self.n_states.append(tuple(tem))

            ind = ind + self.state_dim
            tem = []
            for j in range(ind, ind + self.cost_dim):
                tem.append(ls[j])
            self.costs.append(tuple(tem))

        ##变成set,确定是否是离散数据，然后按照区间划分
        # self.state_set = set(self.states) & set(self.n_states)
        # self.act_set = set(self.acts)
        # self.reward_set = set(self.rewards)
        # self.cost_set = set(self.costs)

        self.state_dict = self.divide(self.state_dim, self.sub_state_range,
                                      self.state_upbound, self.state_lowbound, self.states + self.n_states)

        self.act_dict = self.divide(self.act_dim, self.sub_act_range,
                                    self.act_upbound, self.act_lowbound, self.acts)

        self.reward_dict = self.divide(self.rewd_dim, self.sub_rewd_range,
                                       self.rewd_upbound, self.rewd_lowbound, self.rewards)

        self.cost_dict = self.divide(self.cost_dim, self.sub_cost_range,
                                     self.cost_upbound, self.cost_lowbound, self.costs)

        ##区间处理
        for ls in self.rows:
            ind = 1
            tup = tuple(ls[ind:ind + self.state_dim])
            low, up = self.approximate(tup, self.state_dim, self.sub_state_range, self.state_lowbound,
                                       self.state_upbound)
            appro_state = self.range_transfer(low, up)

            ind = ind + self.state_dim
            tup = tuple(ls[ind:ind + self.act_dim])
            low, up = self.approximate(tup, self.act_dim, self.sub_act_range, self.act_lowbound, self.act_upbound)
            appro_act = self.range_transfer(low, up)

            ind = ind + self.act_dim
            tup = tuple(ls[ind:ind + self.rewd_dim])
            low, up = self.approximate(tup, self.rewd_dim, self.sub_rewd_range, self.rewd_lowbound, self.rewd_upbound)
            appro_rewd = self.range_transfer(low, up)

            ind = ind + self.rewd_dim
            tup = tuple(ls[ind:ind + self.state_dim])
            low, up = self.approximate(tup, self.state_dim, self.sub_state_range, self.state_lowbound,
                                       self.state_upbound)
            appro_nstate = self.range_transfer(low, up)

            ind = ind + self.state_dim
            done = ls[ind]

            ind = ind + 1
            tup = tuple(ls[ind:ind + self.cost_dim])
            low, up = self.approximate(tup, self.cost_dim, self.sub_cost_range, self.cost_lowbound, self.cost_upbound)
            appro_cost = self.range_transfer(low, up)

            self.appro_rows.append([appro_state, appro_act, appro_rewd, appro_nstate, done, appro_cost])

        # 考虑状态的转移
        for ls in self.appro_rows:
            judge = True
            tran = str(tuple([ls[0], ls[3]]))
            for dic in self.state_trans:
                if tran == dic["tran"]:
                    dic["attr"].append([ls[1], ls[2], ls[4], ls[5]])
                    dic["num"] = dic["num"] + 1
                    judge = False
                    break
            if judge:
                self.state_trans.append({"tran": tran, "attr": [ls[1], ls[2], ls[4], ls[5]],
                                         "num": 1, "pro": 0})

        for dic in self.state_trans:
            tot = 0
            for it in self.appro_rows:
                tup = eval(dic["tran"])
                if it[0] == tup[0]:
                    tot = tot + 1

            dic["pro"] = dic["num"] / tot


        # 考虑状态及动作转移
        # for ls in self.appro_rows:
        #     judge = True
        #     tran = str(tuple([ls[0], ls[3], ls[1]]))
        #     for dic in self.state_act_trans:
        #         if tran == dic["tran"]:
        #             dic["attr"].append([ls[2], ls[4], ls[5]])
        #             dic["num"] = dic["num"] + 1
        #             judge = False
        #             break
        #     if judge:
        #         self.state_act_trans.append({"tran": tran, "attr": [ls[2], ls[4], ls[5]],
        #                                      "num": 1, "pro": 0})
        #
        # for dic in self.state_act_trans:
        #     tot = 0
        #     for it in self.appro_rows:
        #         tup = eval(dic["tran"])
        #
        #         if it[0] == tup[0] and it[1] == tup[2]:
        #             tot = tot + 1
        #
        #     dic["pro"] = dic["num"] / tot

            # for dic in self.state_dict:
            #     if dic[]==
            # dic["pro"] = dic["num"] / self.state_dict

    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

    def get_data(self):
        """在self.state_trans中，有tran为字符串,需要转换成为元组"""
        for i in range(len(self.state_trans)):
            self.state_trans[i]["tran"] = ast.literal_eval(self.state_trans[i]["tran"])
        data = []
        last_same_state = self.state_trans[0]["tran"][0]
        line = [self.extract_data(self.state_trans[0])]
        """
        将状态相同的放在一起，同时将 tran相同的动作放在一起
        传出的结果为：状态，次态1，次态对应动作信息1，次态1概率，次态2，次态对应动作信息2，次态概率2...
        """
        for i in range(len(self.state_trans) - 1):
            """状态不同，创建新的一行;否则继续添加"""
            if last_same_state != self.state_trans[i + 1]["tran"][0]:
                data.append(line)
                line = [self.extract_data([self.state_trans[i + 1]])]
                last_same_state = self.state_trans[i + 1]["tran"][0]
            else:
                line.append(self.extract_data([self.state_trans[i + 1]]))
        data.append(line)

        """添加csv存储"""

        return data

    """供get_data()调用，传入一个line，将其变为列表,将这一行中拥有的动作数量放在行的最后，case-to-case->我是真的服了！这个tools是什么鬼输出！"""

    def extract_data(self, data):
        extracted_data = []
        if isinstance(data, tuple) or isinstance(data, list):
            for item in data:
                extracted_data.extend(self.extract_data(item))
        elif isinstance(data, dict):
            for item in data.values():
                extracted_data.extend(self.extract_data(item))
        else:
            extracted_data.append(data)
        return extracted_data

    def write_csv(self, csv_file):
        header1 = ["tran", "attr", "num", "pro"]
        with open(csv_file, 'w', encoding='utf-8', newline='') as file:
            Writer = csv.DictWriter(file, header1)
            Writer.writeheader()
            Writer.writerows(self.state_trans)
        # header2 = ["tran", "attr", "num", "pro"]
        # with open('result2.csv', 'w', encoding='utf-8', newline='') as file:
        #     Writer = csv.DictWriter(file, header2)
        #     Writer.writeheader()
        #     Writer.writerows(self.state_act_trans)

    def divide(self, dim, rge, upbound, lowbound, ls):
        tup = lowbound
        ret_dicts = []

        # 遍历所有网格
        while tup != upbound:
            dict = {"lowbound": tup}
            arr = np.array(tup, dtype="float")
            for i in range(dim):
                arr[i] = min(upbound[i], round(arr[i] + rge[i], math.ceil(math.log10(rge[i]) * -1)))

            dict["upbound"] = tuple(arr)
            dict["bound"] = self.range_transfer(dict["lowbound"], dict["upbound"])
            tup = self.increase(dim, tup, rge, upbound, lowbound)
            dict["elements"] = []
            ret_dicts.append(dict)

        # 分到网格里面
        for item in ls:
            low, up = self.approximate(item, dim, rge, lowbound, upbound)
            for dict in ret_dicts:
                if dict["lowbound"] == tuple(low):
                    dict["elements"].append(item)

        for item in ls:
            dict["bound"] = self.range_transfer(dict["lowbound"], dict["upbound"])
            dict["num"] = len(dict["elements"])
        return ret_dicts

    def approximate(self, tup, dim, rge, lowbound, upbound):
        arr = np.array(tup, dtype='float')
        low = np.zeros(dim, dtype='float')
        up = np.zeros(dim, dtype='float')
        for i in range(len(low)):
            low[i] = max(lowbound[i], round(arr[i] - arr[i] % rge[i], math.ceil(math.log10(rge[i]) * -1)))
            up[i] = min(upbound[i], round(low[i] + rge[i], math.ceil(math.log10(rge[i]) * -1)))
        return low, up

    def range_transfer(self, low, up):
        ret = []
        for i in range(len(low)):
            tup = tuple([low[i], up[i]])
            ret.append(tup)
        return ret

    def increase(self, dim, tup, rge, upbound, lowbound):
        arr = np.array(tup, dtype=float)
        ind = dim - 1
        arr[ind] = round(arr[ind] + rge[ind], math.ceil(math.log10(rge[ind]) * -1))
        while ind > 0:
            if upbound[ind] < arr[ind]:
                arr[ind] = round(lowbound[ind], math.ceil(math.log10(rge[ind]) * -1))
                ind = ind - 1
                arr[ind] = round(arr[ind] + rge[ind], math.ceil(math.log10(rge[ind]) * -1))
            else:
                break

        if upbound[0] < arr[0]:
            return upbound
        tup = tuple(arr)
        return tup


##dict:[lowbound :'',upbound:"",element:[(a,b)],[(a,b)]]

def get_csv_info_new(csv_path, frequency, *columns):
    combined_data = []
    counter = 0  # 计数器变量，用于控制每隔 freq 读取一次数据
    line_number = 0  # 当前行号
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            line_number += 1  # 增加行号计数
            try:
                if counter % frequency == 0:  # 判断是否是需要读取的行
                    data = []
                    for column in columns:
                        """如果是False或者True,报错"""
                        value = float(row[column])
                        data.append(value)
                    combined_data.append(data)
                counter += 1  # 计数器增加1
            except KeyError as e:
                print(f"KeyError at line {line_number}: {e}")
    return combined_data


if __name__ == '__main__':
    data = data_prcss(9, 2, 1, 1, 1, (1e-3, 1e-3), (1e-3,), (1e-3,), (1e-3,), (1e-1, 1e-1), (1e-2,), (1e-2,), (1e-2,),
                      (1, 1), (-1, -1), (1,), (-1,), (2,), (0,), (100,), (0,))
    data.read_in("test.csv")
    data.process()
    # data.write_csv()
    print(data.get_data())

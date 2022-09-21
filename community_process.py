"""
Created on 2022.9.20

@author: Xu Yi

"""


import pandas as pd
import networkx as nx
import csv
from tmoga import TMOGA, evaluation
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_row", 1000)
pd.set_option("display.max_columns", 1000)

'''数据准备，将原始数据与节点id相连接，并去除无用列，保留剩下的列'''


def data_preparation(train_operation_file_path, station_id_file_path):
    d = pd.read_csv(train_operation_file_path)  # 读取列车运行数据
    d2 = pd.read_csv(station_id_file_path)  # 读取站点，编号数据
    d = d.drop_duplicates()  # 去重
    d2 = d2.drop_duplicates()  # 去重
    result = merge_dataFrame(d, d2, on_list=['station_name'])  # 根据station_name一列进行合并
    result.drop(columns=['train_direction',
                         'scheduled_arrival_time', 'scheduled_departure_time',
                         'stop_time', 'actual_arrival_time', 'actual_departure_time',
                         'arrival_delay', 'departure_delay', 'wind', 'weather', 'temperature',
                         'major_holiday', 'province'], inplace=True)  # 丢弃一些没必要的数据
    return result  # 返回剩余列的数据


"""
         date train_number station_name  station_order  station_id
0  2019-10-09           G1          北京南              1          88
1  2019-10-10           G1          北京南              1          88
2  2019-10-11           G1          北京南              1          88
3  2019-10-12           G1          北京南              1          88
"""

'''数据合并'''


def merge_dataFrame(left, right, on_list):
    result = pd.merge(left, right, how='inner', on=on_list)
    return result


'''读取数据，获得时间-站点字典'''


def read_data(result):
    d = result  # 保存result
    t_list = []  # 保存时间
    station_list = []  # 保存各个时间点的站点
    data_list = {}  #
    for i in range(len(d)):
        t = d.iloc[i, 0]  # 获取时间
        if t not in t_list:
            t_list.append(t)  # 当前时间加到list中
            station_id = d.iloc[i, 4]  # 获取id
            station_list.append(station_id)  # 加到当前时间的list中
            data_list[str(t)] = station_list  # 保存到数据表中
            station_list = []  # 将当前时间表置空
        else:
            station_id = d.iloc[i, 4]
            if station_id not in data_list[str(t)]:
                data_list[str(t)].append(station_id)  # 当前数据添加到对应时间列表中
    # 保存到本地备份
    file_name = "time_nodelist.txt"
    file_object = open(file_name, 'w')
    for key in data_list.keys():
        file_object.write(str(key))
        file_object.write("\n")
        file_object.write(str(data_list[key]))
        file_object.write("\n")
    file_object.close()
    # data_list的组成：键为时间，值为该时间的站点
    return data_list


'''
2019-10-09
[88, 419, ...
2019-10-10
[88, 419, ...
2019-10-11
[88, 419, ...
'''

'''获取每个时间片中存在的边'''


def get_dynamic_adjacent_relation(time_nodes_dict, node_distance_filepath, file_name="dynamic_graph_record.csv"):
    node_distance = pd.read_csv(node_distance_filepath)
    file_object = open(file_name, 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(file_object)
    csv_write.writerow(['From', 'To', 'Datetime'])
    for key in time_nodes_dict.keys():
        time = str(key)  # 获取时间
        tndv_list = time_nodes_dict[str(key)]  # time_node_dict_valuelist # 获取当前时间的节点列表
        for i in range(len(node_distance)):  # 遍历当前的边
            if node_distance.iloc[i, 0] in tndv_list and node_distance.iloc[
                i, 1] in tndv_list:  # 如果当前边的fromid和toid都在当前时间节点中
                csv_write.writerow([str(node_distance.iloc[i, 0]), str(node_distance.iloc[i, 1]), str(time)])  # 写入文件
    file_object.close()
    # 返回文件路径
    return file_name


'''
From,To,Datetime
372,141,2019-10-09
141,372,2019-10-09
141,658,2019-10-09
'''

'''按时间排序'''


def sort_DF_according_date(time_out_of_order_time_edges):
    df = pd.read_csv(time_out_of_order_time_edges)
    df.sort_values(by='Datetime', inplace=True)
    return df


'''
,From,To,Datetime
239607,623,658,2019-10-08
237128,60,707,2019-10-08
237127,708,197,2019-10-08
'''

'''几个时间片合并成一个'''


def merge_edgelist_by_date(time_edge_data):
    # time_edge_data.drop(columns=["Unnamed: 0"], inplace=True)
    # date_day = time_edge_data["Datetime"].unique() # 112天 28*4 14*8 16*7
    # 28天的数据合并为一个
    # 2019.10.08——2019.11.5  2019.11.6——2019.12.4  2019.12.5——2020.1.2 2020.1.3——2020.1.27
    # list = ["2019-11-05","2019-11-06","2019-12-04","2019-12-05","2020-01-02","2020-01-03"]
    # for l in list:
    #     row_index = date[date["Datetime"] == l].index.tolist()[0]
    #     print(row_index)
    # row_index = [104440,108170, 212890, 216638, 323970, 327960]
    edge_Frame1 = time_edge_data[:104440].copy()
    edge_Frame2 = time_edge_data[104440:212890].copy()
    edge_Frame3 = time_edge_data[212890:323970].copy()
    edge_Frame4 = time_edge_data[323970:].copy()
    edge_Frame_list = [edge_Frame1, edge_Frame2, edge_Frame3, edge_Frame4]
    i = 1
    for edge_frame in edge_Frame_list:
        edge_frame["Datetime"] = "time{}".format(str(i))
        edge_frame = edge_frame.drop_duplicates(subset=['From', 'To'], keep='first', inplace=False)
        i = i + 1

    new_data = pd.concat([edge_Frame1, edge_Frame2, edge_Frame3, edge_Frame4], axis=0)
    if "Unnamed: 0" in new_data.columns:
        new_data.drop(columns=["Unnamed: 0"], inplace=True)
    return new_data


'''产生动态图序列'''


def generate_graph_list(merge_time_edges):
    d = merge_time_edges
    graph_list = []
    t = d.iloc[0, 2]
    G = nx.Graph()
    for i in range(len(d)):
        current_time = d.iloc[i, 2]
        # print(current_time)
        if current_time == t:
            G.add_edge(d.iloc[i, 0], d.iloc[i, 1])
        else:
            graph_list.append(G)
            G = nx.Graph()
            G.add_edge(d.iloc[i, 0], d.iloc[i, 1])
        t = current_time
    graph_list.append(G)
    # print(graph_list)
    # print(graph_list[0].nodes)
    return graph_list


'''动态图社区检测'''


def dynamic_community_detection(dynamic_network):
    tmoga_model = TMOGA(dynamic_network)
    solutions, solutions_population = tmoga_model.start()
    # for i in range(len(solutions)):
    #     communities = evaluation.parse_locus_solution(solutions[i])
    #     print(communities)
    # {0: [0, 45, ...],...,n: [...]}
    return evaluation.parse_locus_solution(solutions[len(solutions) - 1])


'''获取社区间交互的节点对'''


def get_community_mutual_and_nodes(communities, merge_time_edges):
    edgelist = merge_time_edges
    edgelist = edgelist[323970:]
    node_list_total = []
    communities_list_total = []  # len(edgelist)
    for i in range(len(edgelist)):
        node1 = None
        node2 = None
        community1 = None
        community2 = None
        x = 1
        y = 1
        for key in communities.keys():
            if edgelist.iloc[i, 0] in communities[key]:
                # print("第{}号节点在第{}个社区里".format(edgelist.iloc[i,0],key))
                node1 = edgelist.iloc[i, 0]
                community1 = key
                x = 0
            if edgelist.iloc[i, 1] in communities[key]:
                # print("第{}号节点在第{}个社区里".format(edgelist.iloc[i,1],key))
                node2 = edgelist.iloc[i, 1]
                community2 = key
                y = 0
            if x == 0 and y == 0:
                break
        if community1 != community2:
            node_list = [node1, node2]
            community_list = [community1, community2]
            node_list_total.append(node_list)
            communities_list_total.append(community_list)
    return node_list_total


'''社区间交互的节点对去重'''


def community_mul_unique(commuities, merge_time_edges):
    community_mul = get_community_mutual_and_nodes(commuities, merge_time_edges)
    community_mul = [list(t) for t in set(tuple(_) for _ in community_mul)]
    return community_mul


'''合并数据'''


def time_distance_hebing_data(node_distance_filepath, merge_time_edges):
    # 合并数据
    d = pd.read_csv(node_distance_filepath)
    d2 = merge_time_edges
    d = d.drop_duplicates()
    d2 = d2.drop_duplicates()
    result = merge_dataFrame(d, d2, on_list=['From', 'To'])
    if "Unnamed: 0" in result.columns:
        result.drop(columns=["Unnamed: 0"], inplace=True)
    result.sort_values(by="Datetime", inplace=True)
    return result


'''得到动态加权图'''


def generate_graph_list2(time_distance_edge_data):
    d = time_distance_edge_data
    graph_list = []
    t = d.iloc[0, 3]
    G = nx.Graph()

    for i in range(len(d)):
        current_time = d.iloc[i, 3]
        # print(current_time)
        if current_time == t:
            G.add_edge(d.iloc[i, 0], d.iloc[i, 1], weight=d.iloc[i, 2])
        else:
            graph_list.append(G)
            G = nx.Graph()
            G.add_edge(d.iloc[i, 0], d.iloc[i, 1], weight=d.iloc[i, 2])
        t = current_time
    graph_list.append(G)
    return graph_list


'''获取社区间节点关系'''


def get_node_relation(community_node_list, G1, G2, G3, G4, filename="node_node_relation.csv"):
    file_name = filename
    file_object = open(file_name, 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(file_object)
    csv_write.writerow(['From', 'To', 'relation'])
    i = 0
    j = 0
    for node_pair in community_node_list:
        i = i + 1
        R_12 = find_node_relation_between_adjacent_time_slice(G1, G2, node_pair[0], node_pair[1])
        R_23 = find_node_relation_between_adjacent_time_slice(G2, G3, node_pair[0], node_pair[1])
        R_34 = find_node_relation_between_adjacent_time_slice(G3, G4, node_pair[0], node_pair[1])
        R_24 = (R_23 * 4 + R_34 * 4 - 1 * 2) / 8
        R_13 = (R_12 * 4 + R_23 * 4 - 1 * 2) / 8
        R_14 = (R_13 * 8 + R_24 * 8 - R_23 * 4) / 16

        # print("节点{}，节点{}，相似度{}".format(node_pair[0],node_pair[1],R_14))
        csv_write.writerow([node_pair[0], node_pair[1], R_14])
        j = j + 1
    file_object.close()
    return file_name


'''计算相邻时间片之间两个节点间的关系'''


def find_node_relation_between_adjacent_time_slice(G1, G2, node_a, node_b):
    if node_a in list(G1.nodes) and node_b in list(G1.nodes) and node_a in list(G2.nodes) and node_b in list(G2.nodes):

        A_1 = list(G1.neighbors(node_a))  # 节点a在G1中的邻居
        B_1 = list(G1.neighbors(node_b))  # 节点b在G1中的邻居
        A_2 = list(G2.neighbors(node_a))  # 节点a在G2中的邻居
        B_2 = list(G2.neighbors(node_b))  # 节点b在G2中的邻居

        LCC_G1 = list(set(A_1).intersection(set(B_1)))  # A_1,B_1交集
        LCC_G2 = list(set(A_2).intersection(set(B_2)))  # A_2,B_2交集

        LCC_Union = list(set(LCC_G1).union(set(LCC_G2)))  # LCC_G1,LCC_G2并集
        # print("LCC_Union",LCC_Union)
        LCC_Intersection = list(set(LCC_G1).intersection(set(LCC_G2)))  # LCC_G1,LCC_G2交集
        # print("LCC_Intersection",LCC_Intersection)
        if len(LCC_Union) != 0:
            sum1 = 0
            sum2 = 0
            for node in LCC_Intersection:
                I_x = 0
                I_y = 0
                I_z = 0
                I_p = 0
                if G1.has_edge(node_a, node):
                    I_x = G1.edges[node_a, node]["weight"]
                if G1.has_edge(node_b, node):
                    I_y = G1.edges[node_b, node]["weight"]
                if G2.has_edge(node_a, node):
                    I_z = G2.edges[node_a, node]["weight"]
                if G2.has_edge(node_b, node):
                    I_p = G2.edges[node_b, node]["weight"]
                Ivalue_G1 = I_x * I_y
                Ivalue_G2 = I_z * I_p
                sum1 = sum1 + Ivalue_G1 + Ivalue_G2
            for node in LCC_Union:
                I_x = 0
                I_y = 0
                I_z = 0
                I_p = 0
                if G1.has_edge(node_a, node):
                    I_x = G1.edges[node_a, node]["weight"]
                if G1.has_edge(node_b, node):
                    I_y = G1.edges[node_b, node]["weight"]
                if G2.has_edge(node_a, node):
                    I_z = G2.edges[node_a, node]["weight"]
                if G2.has_edge(node_b, node):
                    I_p = G2.edges[node_b, node]["weight"]
                Uvalue_G1 = I_x * I_y
                Uvalue_G2 = I_z * I_p
                sum2 = sum2 + Uvalue_G1 + Uvalue_G2
            relation_index_between_nodea_nodeb = round(sum1 / sum2, 4)
            return relation_index_between_nodea_nodeb
        else:
            return 0
    else:
        return 0


def choose_community_node(commuities, choose_community_num):
    commuities_node_num = {}
    node_list = []
    current_num = 0
    for k in commuities.keys():
        # print("k:", str(k))
        # print("node_num:", len(commuities[k]))
        commuities_node_num[k] = len(commuities[k])
    test_data_3 = sorted(commuities_node_num.items(),
                         key=lambda x: x[1], reverse=True)
    for i in test_data_3:
        if current_num < choose_community_num:
            key = i[0]
            node = commuities[key]
            node_list = node + node_list
            current_num = current_num + 1
        else:
            break
    print("共计选择{}个社区".format(current_num))
    print("共计选择{}个节点".format(len(node_list)))
    print("选择节点：", node_list)
    return node_list


def change_index(node_list):
    change_index = {}
    value = 0
    for i in node_list:
        change_index[i] = value
        value = value + 1
    return change_index


def get_new_adj(distance, node_list, file_name="new_adj.csv1"):
    file_object = open(file_name, 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(file_object)
    csv_write.writerow(['From', 'To', 'distance'])
    new_index = change_index(node_list)
    for i in range(len(distance)):
        if distance.iloc[i, 0] in node_list and distance.iloc[i, 1] in node_list:
            csv_write.writerow([new_index[distance.iloc[i, 0]], new_index[distance.iloc[i, 1]], distance.iloc[i, 2]])
    file_object.close()
    print("新的邻接矩阵存储在{}中".format(file_name))
    return file_name


def get_new_rel(relation, node_list, file_name="new_rel.csv1"):
    file_object = open(file_name, 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(file_object)
    csv_write.writerow(['From', 'To', 'relation'])
    new_index = change_index(node_list)
    # print(new_index)
    for i in range(len(relation)):
        if relation.iloc[i, 0] in node_list and relation.iloc[i, 1] in node_list:
            csv_write.writerow([new_index[relation.iloc[i, 0]], new_index[relation.iloc[i, 1]], relation.iloc[i, 2]])
    file_object.close()
    print("新的社区关系矩阵存储在{}中".format(file_name))
    return file_name


def get_new_operation_data(operation, node_list, file_name="new_operation_data1.csv"):
    file_object = open(file_name, 'w', encoding='utf-8', newline="")
    csv_write = csv.writer(file_object)
    csv_write.writerow(["station_id", "station_name", "start_time", "end_time", "depart_delay", "arrive_delay"])
    new_index = change_index(node_list)
    # print(new_index)
    for i in range(len(operation)):
        if operation.iloc[i, 0] in node_list:
            csv_write.writerow(
                [new_index[operation.iloc[i, 0]], operation.iloc[i, 1], operation.iloc[i, 2], operation.iloc[i, 3]
                    , operation.iloc[i, 4], operation.iloc[i, 5]])
    file_object.close()
    print("新的列车运行数据存储在{}中".format(file_name))
    return file_name


'''社区获取与社区关系'''


def main(train_operation_file_path, station_id_file_path, node_distance_filepath):
    # * * * * * * * * * * * * 数据准备与预处理* * * * * * * * * * * * #

    train_operation_file_path = train_operation_file_path
    station_id_file_path = station_id_file_path
    node_distance_filepath = node_distance_filepath
    # 数据预处理：运行数据与站点id数据合并
    result = data_preparation(train_operation_file_path, station_id_file_path)
    print("数据预处理完毕!")
    # 读取数据，data_list的组成：键为时间，值为该时间的站点
    data_list = read_data(result)
    print("数据读取完毕!获取时间-站点列表字典完成")
    # 获取每个时间片上的边,是日期乱序的
    time_out_of_order_time_edges = get_dynamic_adjacent_relation(data_list, node_distance_filepath)
    print("每个时间片上的边获取完毕!(乱序),相关文件存储至{}中".format(time_out_of_order_time_edges))
    # 根据日期排列
    time_ordered_time_edges = sort_DF_according_date(time_out_of_order_time_edges)
    print("每个时间片上的边获取完毕!(时间顺序)")
    # 将数据按照时间顺序划分为四组
    merge_time_edges = merge_edgelist_by_date(time_ordered_time_edges)
    print("n个时间片数据融合完毕!")

    # * * * * * * * * * * * * 动态图生成与动态社区检测* * * * * * * * * * * * #

    # 生成动态图序列
    dynamic_network = generate_graph_list(merge_time_edges)
    print("动态图生成完毕!")
    print("\n")
    # 动态图检测
    commuities = dynamic_community_detection(dynamic_network)
    print("动态图社区检测完毕!")

    # * * * * * * * * * * * * 社区间节点交互捕获* * * * * * * * * * * * #

    # 获取社区间交互的节点对并去重 # 678对，339个节点交互，理论上上限为727*727个交互
    node_pair = community_mul_unique(commuities, merge_time_edges)  # 678
    print("社区间交互的节点对获取完毕")
    # 合并数据,将边，时间，权重放在一起
    time_distance_edge_data = time_distance_hebing_data(node_distance_filepath, merge_time_edges)
    print("原始数据边-时间-权重数据合并完毕")
    # 获取动态加权图
    dynamic_network = generate_graph_list2(time_distance_edge_data)
    G1 = dynamic_network[0]
    G2 = dynamic_network[1]
    G3 = dynamic_network[2]
    G4 = dynamic_network[3]
    print("动态加权图获取完毕")
    # 获取社区间的节点亲密度
    result_path = get_node_relation(node_pair, G1, G2, G3, G4)
    print("社区间节点亲密度获取完毕,相关文件存储至{}中".format(result_path))
    return commuities,result_path


'''选一些数据出来'''


def choose_community(community, choose_community_num, distance_path, relation_path, operation_path):
    # 数据太大，选择一部分节点来跑模型，选取标准为社区划分结果中节点较多的
    subG_node_list = choose_community_node(community, choose_community_num)
    distance = pd.read_csv(distance_path)
    relation = pd.read_csv(relation_path)
    operation = pd.read_csv(operation_path)
    return get_new_adj(distance, subG_node_list),get_new_rel(relation, subG_node_list),get_new_operation_data(operation, subG_node_list),len(subG_node_list)


# * * * * * * * * * * * * 社区检测调用接口* * * * * * * * * * * * #
def dynamic_community_detection_and_relation_finding(train_operation_file_path, station_id_file_path,
                                                     node_distance_filepath):
    community,relation_path = main(train_operation_file_path, station_id_file_path, node_distance_filepath)
    return community,relation_path


# * * * * * * * * * * * * 数据选择调用接口* * * * * * * * * * * * #
# def choose_community_to_train(community, choose_community_num, distance_path, relation_path, operation_path):
#     choose_community(community, choose_community_num, distance_path, relation_path, operation_path)


# '''调用'''
# from community_process import dynamic_community_detection_and_relation_finding
#
# train_operation_file_path = "TMOGA-main/dataset/train_delay_with_direction_with_externel_factors_sorted.csv"
# station_id_file_path = "TMOGA-main/dataset/id_station_city.csv"
# node_distance_filepath = "TMOGA-main/dataset/distance.csv"
# distance_path = "TMOGA-main/dataset/distance.csv"
# relation_path = "TMOGA-main/dataset/node_node_relation.csv"
# operation_path = "TMOGA-main/dataset/station_delay_data_2019-10-08_2020-01-27.csv"
#
# community = dynamic_community_detection_and_relation_finding(train_operation_file_path, station_id_file_path,
#                                                              node_distance_filepath)
#
# choose_community_num = 2
# choose_community_to_train((community, choose_community_num, distance_path, relation_path, operation_path))

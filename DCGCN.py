# -*- coding: utf-8 -*-
"""
Created on 2022.9.20

@author: Xu Yi
"""

import os
import shutil
from time import time
from datetime import datetime
import configparser
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import sys
from lib.utils import compute_val_loss, evaluate, predict
from lib.data_preparation import read_and_generate_dataset
from lib.utils import scaled_Laplacian, cheb_polynomial, get_adjacency_matrix, get_community_relation_matrix, \
    Community_AT
from model import DCGCN as model
from community_process import dynamic_community_detection_and_relation_finding, choose_community
sys.path.append(str(os.getcwd()))
sys.path.append(str(os.getcwd()) + '\lib')

# * * * * * * * * * * * * * * * * * * * * 命令行参数读取 * * * * * * * * * * * * * * * * * * * * #

parser = argparse.ArgumentParser()
parser.add_argument('--device', type=str, default='cuda:0', help='')
parser.add_argument('--max_epoch', type=int, default=40, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.0005, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.99, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--length', type=int, default=60, help='Size of temporal : 6')
parser.add_argument("--force", type=str, default=False,
                    help="remove params dir", required=False)
parser.add_argument("--data_name", type=str, default=8,
                    help="the number of data documents", required=False)
# parser.add_argument('--num_point', type=int, default=170,
#                     help='road Point Number [170/307] ', required=False)
parser.add_argument('--decay', type=float, default=0.92, help='decay rate of learning rate [0.97/0.92]')

# * * * * * * * * * * * * * * * * * * * * * 基础配置(1) * * * * * * * * * * * * * * * * * * * * * #

FLAGS = parser.parse_args()
f = FLAGS.data_name
decay = FLAGS.decay
Length = FLAGS.length
batch_size = FLAGS.batch_size  # batch size
epochs = FLAGS.max_epoch  # 最大epoch
learning_rate = FLAGS.learning_rate  # 学习率
optimizer = FLAGS.optimizer  # 优化器
points_per_hour = 1  # 时间粒度为1小时，所以该值为1
num_for_predict = 12  # 预测未来12个时间段的
num_of_weeks = 2
num_of_days = 1
num_of_hours = 2
num_of_features = 2  # 特征数量
merge = False
model_name = "DCGCN"
params_dir = 'experiment_C'
prediction_path = 'DCGCN_prediction_0%s' % f
wdecay = 0.00
device = torch.device(FLAGS.device)
print('Model is %s' % (model_name))

# * * * * * * * * * * * * * * * * * * 动态社区发现、交互系数计算 * * * * * * * * * * * * * * * * * * #

train_operation_file_path = os.getcwd() + '\\community_data\\train_delay_with_direction_with_externel_factors_sorted.csv'
station_id_file_path = os.getcwd() + "\\community_data\\id_station_city.csv"
node_distance_filepath = os.getcwd() + "\\community_data\\distance.csv"
distance_path = os.getcwd() + "\\community_data\\distance.csv"
operation_path = os.getcwd() + "\\community_data\\station_delay_data_2019-10-08_2020-01-27.csv"
# 获取动态社区，社区间交互
community, relation_path = dynamic_community_detection_and_relation_finding(train_operation_file_path,
                                                                            station_id_file_path,
                                                                            node_distance_filepath)
choose_community_num = 5
# 选择一些数据
new_adj_path, new_relation_path, new_operation_path, num_point = choose_community(
    (community, choose_community_num, distance_path, relation_path, operation_path))

# * * * * * * * * * * * * * * * * * * * * * 基础配置(2) * * * * * * * * * * * * * * * * * * * * * #

# num_nodes = FLAGS.num_point
# num_of_vertices=FLAGS.num_point
num_nodes = num_point  # 节点数量
num_of_vertices = num_nodes
adj_filename = new_adj_path  # 邻接矩阵存储文件
community_df_filename = new_relation_path  # 社区间节点关系存储文件
graph_signal_matrix_filename = new_operation_path  # 列车延误数据存储文件 (2688, 324, 2) 2688 = 112天*24小时  时间粒度为1小时 324个站点 2个特征

# * * * * * * * * * * * * * * * * * * 动态社区交互、注意力 * * * * * * * * * * * * * * * * * * #

adj = get_adjacency_matrix(adj_filename, num_nodes)  # 获取邻接矩阵
community_relation_matrix = get_community_relation_matrix(community_df_filename, num_of_vertices)  # 获取社区关系矩阵
adjs = scaled_Laplacian(adj)  # 获取拉普拉斯矩阵
community_relation = Community_AT(community_relation_matrix, num_of_vertices)  # 增加社区自注意力
supports = ((torch.tensor(adjs)).type(torch.float32) + community_relation.type(torch.float32)).to(device)
# supports=(torch.tensor(adjs)).type(torch.float32).to(device)
print(supports.shape)  # 324*324

# * * * * * * * * * * * * * * * * * * * * * 基础配置(3) * * * * * * * * * * * * * * * * * * * * * #

timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
if params_dir != "None":
    params_path = os.path.join(params_dir, model_name)
else:
    params_path = 'params/%s_%s/' % (model_name, timestamp)

# check parameters file
if os.path.exists(params_path) and not FLAGS.force:
    raise SystemExit("Params folder exists! Select a new params path please!")
else:
    if os.path.exists(params_path):
        shutil.rmtree(params_path)
    os.makedirs(params_path)
    print('Create params directory %s' % (params_path))

# * * * * * * * * * * * * * * * * * * * * * 主函数 * * * * * * * * * * * * * * * * * * * * * #

if __name__ == "__main__":
    # read all data from graph signal matrix file
    print("Reading data...")
    # * * * * * * * * * * * * * * * * * * * * * 读取数据 * * * * * * * * * * * * * * * * * * * * * #
    # Input: train / valid  / test : length x 3 x NUM_POINT x 12
    all_data = read_and_generate_dataset(graph_signal_matrix_filename,
                                         num_of_weeks,
                                         num_of_days,
                                         num_of_hours,
                                         num_for_predict,
                                         points_per_hour,
                                         merge)

    # * * * * * * * * * * * * * * * * * * * * * 集合封装 * * * * * * * * * * * * * * * * * * * * * #
    # test set ground truth
    true_value = all_data['test']['target']
    print(true_value.shape)

    # training set data loader
    train_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['train']['week']),
            torch.Tensor(all_data['train']['day']),
            torch.Tensor(all_data['train']['recent']),
            torch.Tensor(all_data['train']['target'])
        ),
        batch_size=batch_size,
        shuffle=True
    )

    # validation set data loader
    val_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['val']['week']),
            torch.Tensor(all_data['val']['day']),
            torch.Tensor(all_data['val']['recent']),
            torch.Tensor(all_data['val']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # testing set data loader
    test_loader = DataLoader(
        TensorDataset(
            torch.Tensor(all_data['test']['week']),
            torch.Tensor(all_data['test']['day']),
            torch.Tensor(all_data['test']['recent']),
            torch.Tensor(all_data['test']['target'])
        ),
        batch_size=batch_size,
        shuffle=False
    )

    # save Z-score mean and std
    stats_data = {}
    for type_ in ['week', 'day', 'recent']:
        stats = all_data['stats'][type_]
        stats_data[type_ + '_mean'] = stats['mean']
        stats_data[type_ + '_std'] = stats['std']
    np.savez_compressed(
        os.path.join(params_path, 'stats_data'),
        **stats_data
    )
    # * * * * * * * * * * * * * * * * * * * * * 实验配置 * * * * * * * * * * * * * * * * * * * * * #
    # loss function MSE
    loss_function = nn.MSELoss()
    # get model's structure
    net = model(c_in=num_of_features, c_out=64,
                num_nodes=num_nodes, week=24,
                day=12, recent=24,
                K=3, Kt=3)
    net.to(device)  # to cuda
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=wdecay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, decay)
    # * * * * * * * * * * * * * * * * * * * * * 计算损失 * * * * * * * * * * * * * * * * * * * * * #
    # calculate origin loss in epoch 0
    compute_val_loss(net, val_loader, loss_function, supports, device, epoch=0)

    # compute testing set MAE, RMSE, MAPE before training
    evaluate(net, test_loader, true_value, supports, device, epoch=0)
    # * * * * * * * * * * * * * * * * * * * * * 正向传播+反向传播 * * * * * * * * * * * * * * * * * * * * * #
    clip = 5
    his_loss = []
    train_time = []
    for epoch in range(1, epochs + 1):
        train_l = []
        start_time_train = time()
        for train_w, train_d, train_r, train_t in train_loader:
            train_w = train_w.to(device)
            train_d = train_d.to(device)
            train_r = train_r.to(device)
            train_t = train_t.to(device)
            net.train()  # train pattern
            optimizer.zero_grad()  # grad to 0

            output, _, _ = net(train_w, train_d, train_r, supports)
            loss = loss_function(output, train_t)
            # backward p
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(net.parameters(), clip)

            # update parameter
            optimizer.step()

            training_loss = loss.item()
            train_l.append(training_loss)
        scheduler.step()
        end_time_train = time()
        train_l = np.mean(train_l)
        print('epoch step: %s, training loss: %.2f, time: %.2fs'
              % (epoch, train_l, end_time_train - start_time_train))
        train_time.append(end_time_train - start_time_train)

        # compute validation loss
        valid_loss = compute_val_loss(net, val_loader, loss_function, supports, device, epoch)
        his_loss.append(valid_loss)

        # evaluate the model on testing set
        evaluate(net, test_loader, true_value, supports, device, epoch)

        # * * * * * * * * * * * * * * * * * * * * * 保存结果 * * * * * * * * * * * * * * * * * * * * * #

        params_filename = os.path.join(params_path,
                                       '%s_epoch_%s_%s.params' % (model_name,
                                                                  epoch, str(round(valid_loss, 2))))
        torch.save(net.state_dict(), params_filename)
        print('save parameters to file: %s' % (params_filename))

    print("Training finished")
    print("Training time/epoch: %.2f secs/epoch" % np.mean(train_time))

    bestid = np.argmin(his_loss)

    print("The valid loss on best model is epoch%s_%s" % (str(bestid + 1), str(round(his_loss[bestid], 4))))
    best_params_filename = os.path.join(params_path,
                                        '%s_epoch_%s_%s.params' % (model_name,
                                                                   str(bestid + 1), str(round(his_loss[bestid], 2))))
    net.load_state_dict(torch.load(best_params_filename))
    start_time_test = time()
    prediction, spatial_at, temporal_at = predict(net, test_loader, supports, device)
    end_time_test = time()
    evaluate(net, test_loader, true_value, supports, device, epoch)
    test_time = np.mean(end_time_test - start_time_test)
    print("Test time: %.2f" % test_time)

    np.savez_compressed(
        os.path.normpath(prediction_path),
        prediction=prediction,
        spatial_at=spatial_at,
        temporal_at=temporal_at,
        ground_truth=all_data['test']['target']
    )

# -*- coding:utf-8 -*-
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
# import mxnet as mx
# import sys
# sys.path.append('.')
from .utils import get_sample_indices
'''
Each npz file contains one key, named "data", 
the shape is (sequence_length, num_of_vertices, num_of_features).(SL,V,F)
'''

def normalization(train, val, test):
    '''
    Parameters
    ----------
    train, val, test: np.ndarray
    训练/验证/测试
    Returns
    ----------
    stats: dict, two keys: mean and std
    返回字典，两个key：平均数和标准差
    train_norm, val_norm, test_norm: np.ndarray,
                                     shape is the same as original
    训练集/验证集/测试集
    '''

    assert train.shape[1:] == val.shape[1:] and val.shape[1:] == test.shape[1:]

    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)

    def normalize(x):
        return (x - mean) / std

    train = (train).transpose(0,2,1,3)
    val = (val).transpose(0,2,1,3)
    test =(test).transpose(0,2,1,3)

    return {'mean': mean, 'std': std}, train, val, test


def read_and_generate_dataset(graph_signal_matrix_filename,
                              num_of_weeks, num_of_days,
                              num_of_hours, num_for_predict,
                              points_per_hour=12, merge=False):
    '''

    读取数据，用于生成训练/验证/测试dataloader

    Parameters
    ----------
    graph_signal_matrix_filename: str, 流量数据存储文件，path of graph signal matrix file
    num_of_weeks, num_of_days, num_of_hours: int
    num_for_predict: int
    points_per_hour: int, default 12, depends on data
    merge: boolean, default False,
           whether to merge training set and validation set to train model
           是否将训练集与验证集合并
    Returns
    ----------
    feature: np.ndarray,
             shape is (num_of_samples, num_of_batches * points_per_hour,
                       num_of_vertices, num_of_features)
    target: np.ndarray,
            shape is (num_of_samples, num_of_vertices, num_for_predict)
    '''
    data_seq = np.load(graph_signal_matrix_filename)['data']
    #print(data_seq.shape) #(2688,89,2)
    # (sequence_length, num_of_vertices, num_of_features) (S,V,F)
    all_samples = []
    for idx in range(data_seq.shape[0]):
        sample = get_sample_indices(data_seq, num_of_weeks, num_of_days,
                                    num_of_hours, idx, num_for_predict,
                                    points_per_hour)
        # print("sample",sample)
        if not sample:
            continue

        week_sample, day_sample, hour_sample, target = sample
        # print("week_sample",week_sample)
        # print("week_sample.shape",week_sample.shape)
        # print(day_sample.shape)
        # print(hour_sample.shape)
        # (12, 89, 2)
        # (12, 89, 2)
        # (36, 89, 2)
        all_samples.append((
            np.expand_dims(week_sample, axis=0).transpose((0, 2, 3, 1)),  # (1,12,89,2)-->(1,89,2,12) (1,V,F,12)
            np.expand_dims(day_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(hour_sample, axis=0).transpose((0, 2, 3, 1)),
            np.expand_dims(target, axis=0).transpose((0, 2, 3, 1))[:, :, 0, :]
        ))
    split_line1 = int(len(all_samples) * 0.6)
    split_line2 = int(len(all_samples) * 0.8)

    if not merge:
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line1])]
    else:
        print('Merge training set and validation set!')
        training_set = [np.concatenate(i, axis=0)
                        for i in zip(*all_samples[:split_line2])]

    validation_set = [np.concatenate(i, axis=0)
                      for i in zip(*all_samples[split_line1: split_line2])]
    testing_set = [np.concatenate(i, axis=0)
                   for i in zip(*all_samples[split_line2:])]
    # print(training_set)
    train_week, train_day, train_hour, train_target = training_set
    val_week, val_day, val_hour, val_target = validation_set
    test_week, test_day, test_hour, test_target = testing_set

    print('training data: week: {}, day: {}, recent: {}, target: {}'.format(
        train_week.shape, train_day.shape,
        train_hour.shape, train_target.shape))
    print('validation data: week: {}, day: {}, recent: {}, target: {}'.format(
        val_week.shape, val_day.shape, val_hour.shape, val_target.shape))
    print('testing data: week: {}, day: {}, recent: {}, target: {}'.format(
        test_week.shape, test_day.shape, test_hour.shape, test_target.shape))

    '''
    training data: week: (1505, 89, 2, 12), day: (1505, 89, 2, 12), recent: (1505, 89, 2, 36), 
                   target: (1505, 89, 12)
    validation data: week: (502, 89, 2, 12), day: (502, 89, 2, 12), recent: (502, 89, 2, 36), 
                   target: (502, 89, 12)
    testing data: week: (502, 89, 2, 12), day: (502, 89, 2, 12), recent: (502, 89, 2, 36), 
                   target: (502, 89, 12)

    '''

    (week_stats, train_week_norm,
     val_week_norm, test_week_norm) = normalization(train_week,
                                                    val_week,
                                                    test_week)

    (day_stats, train_day_norm,
     val_day_norm, test_day_norm) = normalization(train_day,
                                                  val_day,
                                                  test_day)

    (recent_stats, train_recent_norm,
     val_recent_norm, test_recent_norm) = normalization(train_hour,
                                                        val_hour,
                                                        test_hour)

    all_data = {
        'train': {
            'week': train_week_norm,
            'day': train_day_norm,
            'recent': train_recent_norm,
            'target': train_target,
        },
        'val': {
            'week': val_week_norm,
            'day': val_day_norm,
            'recent': val_recent_norm,
            'target': val_target
        },
        'test': {
            'week': test_week_norm,
            'day': test_day_norm,
            'recent': test_recent_norm,
            'target': test_target
        },
        'stats': {
            'week': week_stats,
            'day': day_stats,
            'recent': recent_stats
        }
    }

    return all_data
# if __name__ == '__main__':
#     graph_signal_matrix_filename = r'C:\Users\xuyi\Desktop\DCGCN-master\DCGCN-master\data\PEMS04\pems04.npz'
#     all_data = read_and_generate_dataset(graph_signal_matrix_filename,
#                                          num_of_weeks=2, num_of_days=1,
#                                          num_of_hours=2, num_for_predict=12,
#                                          points_per_hour=12, merge=False)
#     print(all_data)
#     # print(all)
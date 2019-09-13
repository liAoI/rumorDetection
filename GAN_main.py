'''
    谣言检测的GAN模型：
    通过seq2seq将谣言伪装成非谣言，非谣言伪装成谣言

'''
from utils import IndexOfData,Visualizer
import models

import os
from tqdm import tqdm,trange
import json
import subprocess
import torch
import numpy as np
from torch.utils import data


def my_collate(batch):
    '''
                    batch里面长度补齐实现策略

                    将batch按照x的长度进行排序

                    按照句子的最长长度来补齐
                    2019年9月9日，这里修改一下，不按最长补齐，按以下策略：
                    句子平均值低于2000的按平均值补齐，否则按2000补齐
                '''
    batch.sort(key=lambda data:len(data[0]),reverse=True)
    data_len_x = [len(x[0]) for x in batch]
    data_len_y = [len(x[1]) for x in batch]
    n_words = batch[0][2]

    aver = np.mean(data_len_x)
    aver = int(aver)  # 这里采用强制转换，将平均值转为int型

    if aver < 500:
        pad_col_x = [aver - x for x in data_len_x]
        pad_col_y = [max(data_len_y) - y for y in data_len_y]
        data_x = []
        data_y = []
        for x_cols, y_cols, data ,i in zip(pad_col_x, pad_col_y, batch,range(len(data_len_x))):
            X = np.array(data[0])
            Y = np.array(data[1])
            Y = np.pad(Y, (0, y_cols), 'constant')
            if x_cols > 0 or x_cols == 0:
                X = np.pad(X, (0, x_cols), 'constant')
            else:
                X = X[0:aver]
                data_len_x[i] = aver

            data_x.append(X)
            data_y.append(Y)
    else:
        pad_col_x = [500 - x for x in data_len_x]
        pad_col_y = [max(data_len_y) - y for y in data_len_y]
        data_x = []
        data_y = []
        for x_cols, y_cols, data, i in zip(pad_col_x, pad_col_y, batch, range(len(data_len_x))):
            X = np.array(data[0])
            Y = np.array(data[1])
            Y = np.pad(Y, (0, y_cols), 'constant')
            if x_cols > 0 or x_cols == 0:
                X = np.pad(X, (0, x_cols), 'constant')
            else:
                X = X[0:500]
                data_len_x[i] = 500

            data_x.append(X)
            data_y.append(Y)

    X = torch.LongTensor(data_x)
    Y = torch.FloatTensor(data_y)
    return X, Y, data_len_x, n_words

# def my_collate(batch):
#     '''
#                     batch里面长度补齐实现策略
#
#                     将batch按照x的长度进行排序
#
#                     按照句子的最长长度来补齐
#
#                 '''
#     batch.sort(key=lambda data:len(data[0]),reverse=True)
#     data_len_x = [len(x[0]) for x in batch]
#     data_len_y = [len(x[1]) for x in batch]
#     n_words = batch[0][2]
#     pad_col_x = [max(data_len_x) - x for x in data_len_x]
#     pad_col_y = [max(data_len_y) - y for y in data_len_y]
#     data_x = []
#     data_y = []
#     for x_cols, y_cols, data in zip(pad_col_x, pad_col_y, batch):
#         X = np.array(data[0])
#         X = np.pad(X, (0, x_cols), 'constant')
#         Y = np.array(data[1])
#         Y = np.pad(Y, (0, y_cols), 'constant')
#
#         data_x.append(X)
#         data_y.append(Y)
#
#     X = torch.LongTensor(data_x)
#     Y = torch.FloatTensor(data_y)
#     return X, Y, data_len_x, n_words



def divide_RandNR(dir):
    # 获取数据清单
    flist = []
    label = None
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                flist.append(os.path.join(root, file))

    for filename in tqdm(flist):
        with open(filename, 'r', encoding='utf-8') as f:
            fjson = json.load(f)
            s = fjson['label']
            label = s
        if label == '1':
            subprocess.call('cp ' + filename + ' ' + dir + '/Rumor/', shell=True)
        else:
            subprocess.call('cp ' + filename + ' ' + dir + '/Non-rumor/', shell=True)





if __name__ == '__main__':
    #将谣言分到Rumor和Non-rumor文件夹中
    # divide_RandNR('/root/PycharmProjects/majing-rumor-rnn/RumorDetect')



    n_words = np.load('./n_words.npy',allow_pickle=True).item()
    Trainmodel = models.GAN_train_weibo(n_words)

    ndatas = IndexOfData(dir='./Non-rumor')
    rdatas = IndexOfData(dir='./Rumor')
    train_loader_n = data.DataLoader(ndatas, batch_size=2, num_workers=0, shuffle=False, collate_fn=my_collate)
    train_loader_r = data.DataLoader(rdatas, batch_size=2, num_workers=0, shuffle=False, collate_fn=my_collate)

    #test data
    tndatas = IndexOfData(Train=False,dir='./Non-rumor')
    tdatas = IndexOfData(Train=False,dir='./Rumor')

    test_loader_n = data.DataLoader(tndatas, batch_size=2, num_workers=0, shuffle=False, collate_fn=my_collate)
    test_loader_r = data.DataLoader(tdatas, batch_size=2, num_workers=0, shuffle=False, collate_fn=my_collate)

    vis = Visualizer(env='GAN',port = 8097)

    Trainmodel.forward(train_loader_n,train_loader_r,test_loader_n,test_loader_r,vis,120)



    # for i, trainset in enumerate(train_loader):
    #     X, Y, data_len_x, n_words = trainset
    #     print(X)
    #     print(Y)
    #     print(data_len_x)
    #     print(n_words)
    #     break
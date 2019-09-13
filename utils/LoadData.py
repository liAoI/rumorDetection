from torch.utils import data
import os
from gensim.models import word2vec
import torch
import json
import re
import numpy as np
from tqdm import tqdm

class ShipDataset(data.Dataset):
    def __init__(self,vali = 1,Train = True,dir='./originaldata'):
        super(ShipDataset, self).__init__()
        self.train = []
        self.vaild = []

        #获取数据清单
        flist = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    flist.append(os.path.join(root, file))

        flist = [flist[i:i + len(flist)//5] for i in range(0, len(flist), len(flist)//5)]

        for i in range(len(flist)):
            if i == vali:
                self.vaild = flist[i]
            else:
                self.train +=flist[i]

        self.ModeTrain = Train

        self.MO = word2vec.Word2Vec.load('./Word2Vec/originalSize20WordModel')

    def __getitem__(self, item):
        X = []
        Y = None
        if self.ModeTrain:
            filename = self.train[item]
        else:
            filename = self.vaild[item]

        with open(filename, 'r', encoding='utf-8') as f:
            fjson = json.load(f)
        for i in fjson['text']:
            try:
                X.append(self.MO[i])
            except (KeyError):
                # logger.info(i +'没有向量化！')
                continue
        X = torch.tensor(X)  # 这里由于每条句子长度不一致，导致无法封装到一个batch里，所以才设置取前30行
        Y = int(fjson['label'])
        # Y = torch.tensor(int(fjson['label'])).float()
        if Y == 0 :    #非谣言
            # Y=torch.tensor(Y)
            Y = torch.tensor([0.0,1.0])
        else:
            Y = torch.tensor([1.0,0.0])
        # Y = torch.tensor(Y)
        return X,Y

    def __len__(self):
    #返回数据的数量
        if self.ModeTrain:
            return len(self.train)
        else:
            return len(self.vaild)

'''
    下面是直接用单词索引打包成batch
'''

class IndexOfData(data.Dataset):
    def __init__(self, vali=1, Train=True, dir='./originaldata'):
        super(IndexOfData, self).__init__()
        self.train = []
        self.vaild = []

        self.Word2Index = {'SOS': 0, 'EOS': 1}
        self.index2word = {}
        self.n_words = 2
        self.word2count = {}

        # 获取数据清单
        flist = []
        for root, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    flist.append(os.path.join(root, file))

        if os.path.exists('word2index.npy'):
            self.Word2Index = np.load('word2index.npy', allow_pickle=True).item()
            self.index2word = np.load('index2word.npy', allow_pickle=True).item()
            self.n_words = np.load('n_words.npy', allow_pickle=True).item()

        else:
            self.CreateWord2index(flist)
            np.save('word2index.npy', self.Word2Index)
            np.save('index2word.npy', self.index2word)
            np.save('n_words.npy', self.n_words)


        flist = [flist[i:i + len(flist) // 5] for i in range(0, len(flist), len(flist) // 5)]


        for i in range(len(flist)):
            if i == vali:
                self.vaild = flist[i]
            else:
                self.train += flist[i]
        self.ModeTrain = Train

    def __getitem__(self, item):
        X = []
        Y = None
        if self.ModeTrain:
            filename = self.train[item]
        else:
            filename = self.vaild[item]

        with open(filename, 'r', encoding='utf-8') as f:
            fjson = json.load(f)
            s = self.CleanStr(fjson['text'])
        for i in s:
            try:
                X.append(self.Word2Index[i])
            except (KeyError):
                # logger.info(i +'没有向量化！')
                continue
        Y = int(fjson['label'])
        # Y = torch.tensor(int(fjson['label'])).float()
        if Y == 0:  # 非谣言
            # Y=torch.tensor(Y)
            Y = torch.tensor([0.0, 1.0])
        else:
            Y = torch.tensor([1.0, 0.0])
        return X, Y,self.n_words

    def __len__(self):
        # 返回数据的数量
        if self.ModeTrain:
            return len(self.train)
        else:
            return len(self.vaild)

    def CreateWord2index(self, flist):

        print('------正在构建词典------')
        for filename in tqdm(flist):
            with open(filename, 'r', encoding='utf-8') as f:
                fjson = json.load(f)
                s = fjson['label']
                s = self.CleanStr(s)
                for word in s:
                    if word not in self.Word2Index:
                        self.Word2Index[word] = self.n_words
                        self.word2count[word] = 1
                        self.n_words += 1
                    else:
                        self.word2count[word] += 1
        print('------词典索引构建成功------')


    def CleanStr(self, s):
        s = re.sub(r'([-【】！、。，？“”()（）.!?])', r'', s)  #
        s = s.split(' ')  # 按照空格将词分开
        s = list(filter(None, s))  # 去掉空字符
        return s  # 返回的是list
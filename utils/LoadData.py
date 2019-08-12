from torch.utils import data
import os
from gensim.models import word2vec
import torch
import json

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


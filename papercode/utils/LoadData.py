# coding=utf-8
'''
PAD 0   补全
SOS 1   句子开头
EOS 2   句子结尾
'''
import re,os,json
import numpy as np
from torch.utils import data
class IndexOfData(data.Dataset):
    def __init__(self, vali=1, Train=True, dir='Paper/datasets/pytorchrumor'):
        super(IndexOfData, self).__init__()
        self.train = []
        self.vaild = []

        self.Word2Index = {'SOS': 1, 'EOS': 2}
        self.index2word = {1:'SOS', 2:'EOS'}
        self.n_words = 2
        self.word2count = {}

        # 获取数据清单
        flist = {}
        for root, dirs, files in os.walk(dir):
            for file in files:
                if os.path.splitext(file)[1] == '.json':
                    flist[os.path.join(root, file)] = os.path.getsize(os.path.join(root, file))

        sort_result = sorted(flist.items(), key=lambda item: item[1], reverse=False)
        flist = [name_size[0] for name_size in sort_result]

        self.Word2Index = np.load('../preprocess/word2index.npy', allow_pickle=True).item()
        self.index2word = np.load('../preprocess/index2word.npy', allow_pickle=True).item()
        self.n_words = np.load('../preprocess/n_words.npy', allow_pickle=True).item()
        self.word2count = np.load('../preprocess/word2count.npy', allow_pickle=True).item()

        flist = [flist[i:i + len(flist) // 5] for i in range(0, len(flist), len(flist) // 5)]

        for i in range(len(flist)):
            if i == vali:
                self.vaild = flist[i]
            else:
                self.train += flist[i]
        self.ModeTrain = Train

    def __getitem__(self, item):
        X = []

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

        X.insert(0,1) #句子开始符号
        X.append(2)#句子结束符号
        return X, Y

    def __len__(self):
        # 返回数据的数量
        if self.ModeTrain:
            return len(self.train)
        else:
            return len(self.vaild)


    def CleanStr(self, s):
        s = re.sub(r'([-【】！、。，？“”()（）.!?])', r'', s)  #
        s = s.split(' ')  # 按照空格将词分开
        s = list(filter(None, s))  # 去掉空字符
        return s  # 返回的是list
'''
201907 先将文本通过卷积提取特征，然后进行rnn
实现思路请看论文：http://gxbwk.njournal.sdu.edu.cn/CN/Y2019/V49/I2/102
'''
import torch
import torch.nn as nn
from .BasicModule import BasicModule
class RNNModel(BasicModule):
    def __init__(self, input_size, hidden_size, n_layers, lstm=True, GPU=True):
        super(RNNModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = lstm
        self.hidden_size = hidden_size
        self.gpu = GPU
        if self.gpu == True:
            # 加卷积层  in_channels=1 out_channels=1 kernel_size=(1,20) 20是因为word是1行20维的向量 stride=1 padding=0 dilation=1
            self.cnn_1 = nn.Conv2d(1, 60, (2, 20), 1, 0).cuda(device=0)
            self.cnn_2 = nn.Conv2d(1, 60, (3, 20), 1, 0).cuda(device=0)
            self.cnn_3 = nn.Conv2d(1, 60, (4, 20), 1, 0).cuda(device=0)
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2,
                               bidirectional=True).cuda(device=0)
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True, dropout=0.4).cuda(device=0)
            # self.linear = nn.Linear(self.hidden_size,2)  #二分类，最后结果[0,1] [1,0]
            self.layer = nn.Sequential(nn.Linear(self.hidden_size * 2, 80), nn.BatchNorm1d(80), nn.ReLU(),
                                       nn.Linear(80, 30), nn.BatchNorm1d(30), nn.ReLU(),
                                       nn.Linear(30, 2), nn.BatchNorm1d(2), nn.Sigmoid()
                                       ).cuda(device=0)

    def forward(self, input, state=None):
        batch, words, vecsize = input.size()  # 第一个是batchsize 第二个是单词个数 第三个是词向量的维度
        input = input.view(batch, 1, words, vecsize)  # 这样做的目的是将一个句子当做一副通道数为1的图片，然后进行卷积
        input_1 = self.cnn_1(input)
        input_2 = self.cnn_2(input)
        input_3 = self.cnn_3(input)
        # 按最小words维度进行截取
        _, _, words1, _ = input_1.size()
        _, _, words2, _ = input_2.size()
        _, _, words3, _ = input_3.size()
        words = min(words1, words2, words3)
        input_1 = input_1[:, :, 0:words, :]
        input_2 = input_2[:, :, 0:words, :]
        input_3 = input_3[:, :, 0:words, :]
        input = torch.cat([input_1, input_2, input_3], dim=1)
        input = torch.squeeze(input)
        input = input.permute(0,2,1)
        # 这里按照同一位置卷积的结果拼接在一起
        if self.gpu == True:
            if self.lstm == True:
                if state is None:
                    h = torch.randn(self.n_layers * 2, batch, self.hidden_size).cuda(device=0).float()
                    c = torch.randn(self.n_layers * 2, batch, self.hidden_size).cuda(device=0).float()
                else:
                    h, c = state

                # output [batchsize,time,hidden_size]
                output, state = self.rnn(input, (h, c))
            else:
                if state is None:
                    state = torch.randn(self.n_layers, batch, self.hidden_size).cuda(device=0).float()
                output, state = self.gru(input, state)

        # 最后输出结果
        output = self.layer(output[:, -1, :])
        return output, state

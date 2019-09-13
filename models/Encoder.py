import torch.nn.functional as F
import torch
import torch.nn as nn
from .BasicModule import BasicModule
'''
    20190828
'''
class encoderModel(BasicModule):
    def __init__(self,n_layers=2,input_size=10,hidden_size=10):
        super(encoderModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.embeding = nn.Embedding(n_words,input_size,padding_idx=0)

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.4,bidirectional=False)

    def forward(self,input,len_x,state=None):
        # 自己给自己挖坑，这里输入的input是单词的index，我现在想用可训练的词向量，也就是nn.embeding
        # input = self.embeding (input)

        #这里输入的input是已经词向量化好的数据
        batch, words, vecsize = input.size()   #第一个是batchsize 第二个是单词个数 第三个是词向量的维度


        input_x = torch.nn.utils.rnn.pack_padded_sequence(input, len_x, batch_first=True)

        if state is None:
            h = torch.randn(self.n_layers, batch, self.hidden_size).float()
            c = torch.randn(self.n_layers, batch, self.hidden_size).float()
        else:
            h, c = state


        # output [batchsize,time,hidden_size]
        output, state = self.rnn(input_x, (h, c))
        output,_ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        #最后输出结果
        # output = output[:, -1, :]
        output = F.log_softmax(output,dim=1)
        return output,state

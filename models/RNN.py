from .BasicModule import BasicModule
import torch
import torch.nn as nn

class rnn(BasicModule):
    def __init__(self,input_size,hidden_size,n_layers,lstm=True,GPU=True):
        super(rnn, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.lstm = lstm
        self.hidden_size = hidden_size
        self.gpu = GPU
        if self.gpu == True:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True,dropout=0.4).cuda()
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,dropout=0.4).cuda()
            # self.linear = nn.Linear(self.hidden_size,2)  #二分类，最后结果[0,1] [1,0]
            self.layer = nn.Sequential(nn.Linear(self.hidden_size, 50), nn.BatchNorm1d(50), nn.ReLU(),
                                       nn.Linear(50, 15), nn.BatchNorm1d(15), nn.ReLU(),
                                       nn.Linear(15, 2), nn.BatchNorm1d(2), nn.Sigmoid()
                                       ).cuda()
        else:
            self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True,dropout=0.4)
            self.gru = nn.GRU(input_size, hidden_size, n_layers, batch_first=True,dropout=0.4)
            self.layer = nn.Sequential(nn.Linear(self.hidden_size, 50), nn.BatchNorm1d(50),nn.ReLU(),
                                        nn.Linear(50,15),nn.BatchNorm1d(15),nn.ReLU(),
                                       nn.Linear(15, 2), nn.BatchNorm1d(2), nn.Sigmoid()
                                       )

    def forward(self,input,state=None):
        batch, _, _ = input.size()
        if self.gpu == True:
            if self.lstm == True:
                if state is None:
                    h = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                    c = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                else:
                    h, c = state

                # output [batchsize,time,hidden_size]
                output, state = self.rnn(input, (h, c))
            else:
                if state is None:
                    state = torch.randn(self.n_layers, batch, self.hidden_size).cuda().float()
                output, state = self.gru(input, state)
        else:
            if self.lstm == True:
                if state is None:
                    h = torch.randn(self.n_layers, batch, self.hidden_size).float()
                    c = torch.randn(self.n_layers, batch, self.hidden_size).float()
                else:
                    h, c = state

                # output [batchsize,time,hidden_size]
                output, state = self.rnn(input, (h, c))
            else:
                if state is None:
                    state = torch.randn(self.n_layers, batch, self.hidden_size).float()
                output, state = self.gru(input, state)
        #最后输出结果
        output = self.layer(output[:, -1, :])
        return output,state
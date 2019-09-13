import torch
import torch.nn as nn
import torch.nn.functional as F

'''
    一个解码器单元结构：
    解码器：使用两层lstm，加上注意力机制
    初始的隐藏状态是编码器最后LSTM的隐藏状态
    开始输入值为SOS_token = 0，结束值为EOS_token = 1。
    解码器的输入值为前一个时刻的输出和由attention权重算出的编码器上下文语义的拼接

    attention： score(State,output) = W*state*output  其中W为要学习的权重
    这里用一个线性层来实现

'''


class ArrdecoderModel(nn.Module):
    def __init__(self,n_layers=2,input_size=10,hidden_size=10):
        super(ArrdecoderModel, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size  #由于直接用编码器的state初始化解码器，所以这里的hidden_size和编码器的相等
        self.hidden_size = hidden_size

        if input_size != hidden_size:
            raise RuntimeError('解码器的前一个输出即为后一个时刻的输入，所以inputSize应该与hiddenSize一致！')

        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.2)
        self.att_line = nn.Linear(hidden_size*2,1)
        self.input_line = nn.Linear(hidden_size + input_size, input_size)

        # self.out = nn.Linear(hidden_size,n_words)

    def forward(self,input,state,encoder_output):


        #解码器的第一个输入是为SOS_token = 0,如果输出为EOS_token = 1
        batch, words, hidden_size = encoder_output.size()   # output [batchsize,time,hidden_size]

        if state is None:
            raise RuntimeError('解码器的state为空，请将编码器的state或者前一个时刻的state传进来！')

        '''加入attention
            将前一个状态输出的state和encoder_output进行F(X)，得到对encoder_output各输出的权重
            对encoder_output进行加权求和，得到这一时刻的输入
        '''
        #这里取最后一层的state来进行计算,将state里的h点乘c作为状态，其实也可以直接拿h或者c
        #这样得到的att_state = [batch,hidden]
        att_state = torch.mul(state[0][-1],state[1][-1])
        #初始化权重
        weigth = torch.zeros(batch, 1)
        #计算权重 这里将编码器每一个时间点的输出与att_state进行全连接得到一个权重值
        for word in range(words):
            l = torch.cat((encoder_output[:, word, :], att_state), dim=1)
            w = self.att_line(l)
            weigth = torch.cat((weigth, w), dim=1)
        weigth = F.softmax(weigth[:, 0:-1],dim=1)  #去除第一列然后进行softmax得到权重

        #将权重与encoder_output相乘
        weigth = weigth.unsqueeze(1)

        x = torch.bmm(weigth, encoder_output)  #[batch,time=1,hidden_size]

        x = torch.cat((x,input),dim=2)  #前一时刻的输出和attention计算出的上下文语义拼接，并进行全连接成[batch,time=1,hidden]
        input = self.input_line(x)


        # output [batchsize,time,hidden_size]
        output, state = self.rnn(input, state)

        # output = self.out(output.squeeze()).unsqueeze(1)

        # output = F.log_softmax(output,dim=2)
        return output,state
import torch
import torch.nn.functional as F
import torch.nn as nn
class AttnDecoder(nn.Module):
    def __init__(self,atten_model,embedding,hiddenSize,output_size,n_layers=1,dropout = 0.1):
        super(AttnDecoder, self).__init__()
        self.atten_model = atten_model
        self.embedding = embedding
        self.hiddenSize = hiddenSize
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.gru = nn.GRU(hiddenSize,hiddenSize,n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True)
        self.att = Attn(atten_model,hiddenSize)
        self.concat = nn.Linear(hiddenSize*2,hiddenSize)
        self.out = nn.Linear(hiddenSize,output_size)
    def forward(self,seq_in,state,encoder_output):
        # '''
        #         Bahdanau等（2015）的方法：将前一隐藏层输出与编码器的输出运算得到权重，然后
        #         将权重乘以编码器的输出并与当前输入进行concat连接得到当前输入，经过当前神经元运算得到
        #         下一个神经元的输入
        #         :param seq_in: 输入序列
        #         :param state: 前一神经元的隐藏层
        #         :param encoder_output: 编码器的输出
        #         :return:
        #         '''
        # embedded = self.embedding(seq_in)
        #
        # onelayerhidden = state[0, :, :].unsqueeze(1)  # batchsize,layer,hiddensize
        # # encoder_output = encoder_output.permute(0,2,1) #batchsize,vocsize,time
        # # 这里计算权重是由解码器的上一时刻隐藏层与编码器的所有输出运算得出
        # atten_weights = self.att(onelayerhidden, encoder_output)
        # # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        # context = atten_weights.bmm(encoder_output)
        #
        # concat_input = torch.cat((embedded, context), 2)
        # input = self.concat(concat_input)
        #
        # output, hidden = self.gru(input, state)
        #
        # output = self.out(output.squeeze(1))
        # output = F.softmax(output, dim=1)
        #
        # return output, hidden, atten_weights

        '''
        Luong 等（2015）使用当前神经单元的输出与编码器的所有输出运算得到权重，将
        权重与编码器输出相乘再与当前单元的输出进行concat操作得到下一个神经元的输入
        :param seq_in:
        :param state:
        :param encoder_output:
        :return:
        '''
        embedded = self.embedding(seq_in)

        output,hidden = self.gru(embedded,state)
        # onelayerhidden = hidden[0,:,:].unsqueeze(1)  #batchsize,layer,hiddensize
        # encoder_output = encoder_output.permute(0,2,1) #batchsize,vocsize,time
        #这里计算权重是由解码器的上一时刻隐藏层与编码器的所有输出运算得出
        atten_weights = self.att(output,encoder_output)
        # 将注意力权重乘以编码器输出以获得新的“加权和”上下文向量
        context = atten_weights.bmm(encoder_output)

        output = output.squeeze(1)
        context = context.squeeze(1)
        concat_input = torch.cat((output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input))
        # 使用Luong的公式6预测下一个单词
        output = self.out(concat_output)
        output = F.softmax(output, dim=-1)
        # 返回输出和在最终隐藏状态
        return output, hidden,atten_weights




class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = torch.nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = torch.nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # 根据给定的方法计算注意力（能量）
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)
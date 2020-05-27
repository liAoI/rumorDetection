import torch as t
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,hiddenSize,embedding,n_layers = 1,dropout=0):
        super(Encoder, self).__init__()
        self.n_layers = n_layers
        self.hiddenSize = hiddenSize
        self.embedding = embedding
        #词向量和隐藏层向量都是hiddenSize
        self.gru = nn.GRU(hiddenSize,hiddenSize,n_layers,
                          dropout=(0 if n_layers == 1 else dropout),
                          batch_first=True,bidirectional=True)
        self.concat = nn.Linear(hiddenSize * 2,hiddenSize)
    def forward(self,seq_in,lengths_seq_in,hidden=None):
        embedded = self.embedding(seq_in)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths_seq_in,batch_first=True)
        outputs, hidden = self.gru(packed, hidden)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs,batch_first=True)
        #这里把正向和反向gru得到的相加
        outputs = self.concat(outputs)
        return outputs,hidden
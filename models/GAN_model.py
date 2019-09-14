import torch
import torch.nn.functional as F
import torch.nn as nn
from .BasicModule import BasicModule
from .Encoder import encoderModel
from .Decoder import ArrdecoderModel
import tqdm


class D(BasicModule):
    def __init__(self, n_layers=2, input_size=10, hidden_size=10):
        super(D, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size


        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=0.4, bidirectional=False)
        self.layer = nn.Linear(self.hidden_size,2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input, len_x, state=None):
        # 自己给自己挖坑，这里输入的input是单词的index，我现在想用可训练的词向量，也就是nn.embeding
        # input = self.embeding(input)

        # 这里输入的input是已经词向量化好的数据
        batch, words, vecsize = input.size()  # 第一个是batchsize 第二个是单词个数 第三个是词向量的维度

        input_x = torch.nn.utils.rnn.pack_padded_sequence(input, len_x, batch_first=True)

        if state is None:
            h = torch.randn(self.n_layers, batch, self.hidden_size).float()
            c = torch.randn(self.n_layers, batch, self.hidden_size).float()
        else:
            h, c = state

        # output [batchsize,time,hidden_size]
        output, state = self.rnn(input_x, (h, c))
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        # 构造torch.gather 的 index
        index = torch.ones((batch, 1, self.hidden_size), dtype=torch.long, requires_grad=False)
        for i in range(len(len_x)):
            index[i, 0, :] = len_x[i] - 1

        output = torch.gather(output, 1, index)
        output = self.layer(output[:, -1, :])
        output = self.sigmoid(output)
        return output, state

class Generator_NR(BasicModule): #定义 Non-rumor -> Rumor
    def __init__(self):
        super(Generator_NR, self).__init__()
        self.encoder = encoderModel()
        self.decoder = ArrdecoderModel()
        print('Generator_NR 编码器',self.encoder,sep='---->')
        print('Generator_NR 解码器', self.decoder, sep='---->')
    def forward(self, input,len_x):
        target_length = input.size(1)

        encoder_outputs, state = self.encoder(input, len_x)
        # 这里定义解码器的开始和结束标志
        decoder_input = torch.zeros((encoder_outputs.size(0), 1, encoder_outputs.size(2))).float()
        output = torch.ones_like(encoder_outputs)
        # END_token = torch.ones(encoder_outputs.size(0)).long()

        for i in range(target_length):
            decoder_output, state = self.decoder(decoder_input, state, encoder_outputs)

            output[:,i,:] = decoder_output[:,0,:]

            decoder_input = decoder_output
            #
            # if torch.equal(decoder_end, END_token):
            #     break
        return output


class Generator_RN(BasicModule):  # 定义 Rumor -> Non-rumor
    def __init__(self):
        super(Generator_RN, self).__init__()
        self.encoder = encoderModel()
        self.decoder = ArrdecoderModel()
        print('Generator_RN 编码器', self.encoder, sep='---->')
        print('Generator_RN 解码器', self.decoder, sep='---->')

    def forward(self, input, len_x):
        target_length = input.size(1)

        encoder_outputs, state = self.encoder(input, len_x)
        # 这里定义解码器的开始和结束标志
        decoder_input = torch.zeros((encoder_outputs.size(0), 1, encoder_outputs.size(2))).float()
        output = torch.ones_like(encoder_outputs)
        # END_token = torch.ones(encoder_outputs.size(0)).long()

        for i in range(target_length):
            decoder_output, state = self.decoder(decoder_input, state, encoder_outputs)


            decoder_input = decoder_output

            output[:,i,:] = decoder_output[:,0,:]

        return output

class GAN_train_weibo:
    def __init__(self,n_words,input_size=10):
        super(GAN_train_weibo, self).__init__()
        self.G_rn = Generator_RN()
        self.G_nr = Generator_NR()

        self.D = D()
        print('GAN MODEL 构建成功')
        print('伪装器 谣言向非谣言',self.G_rn,sep='---->')
        print('伪装器 非谣言向谣言',self.G_nr,sep='---->')
        print('判别器',self.D,sep='---->')

        self.criterion = nn.BCELoss()
        #词向量层
        self.embeding = nn.Embedding(n_words, input_size, padding_idx=0)

        self.G_rn_optimizer = torch.optim.Adam(self.G_rn.parameters(), lr=0.001)
        self.G_nr_optimizer = torch.optim.Adam(self.G_nr.parameters(), lr=0.001)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(),lr=0.001)

        self.loss_grn = 0
        self.loss_gnr = 0
        self.loss_d = 0

    def neg_label(self,y):
        b = torch.ones_like(y)
        result = b + torch.neg(y)

        return result

    def update_G_RN(self,input_r,y,len_x):
        print('-----------------update generator r -> n------------------')
        '''训练伪装器 G_RN'''
        X_rn = self.G_rn(input_r,len_x)

        dgrn,_ = self.D(X_rn,len_x)

        X_rnr = self.G_nr(X_rn,len_x)



        loss = self.criterion(dgrn,self.neg_label(y)) + 0.02 * self.caculate_cmp(input_r,X_rnr)

        self.loss_grn = loss

        print('update generator r -> n', loss)

        self.G_nr.zero_grad()
        loss.backward(retain_graph=True)

    def update_G_NR(self,input_n,y,len_x):
        print('-----------------update generator n -> r------------------')
        '''训练伪装器 G_NR'''
        X_nr = self.G_nr(input_n,len_x)

        dgnr,_ = self.D(X_nr,len_x)

        X_nrn = self.G_rn(X_nr,len_x)

        loss = self.criterion(dgnr,self.neg_label(y)) + 0.02 * self.caculate_cmp(input_n,X_nrn)

        self.loss_gnr = loss

        print('update generator n -> r',loss)

        self.G_rn.zero_grad()
        loss.backward(retain_graph=True)

    def update_D(self,flag,input,y,len_x):
        print('-----------------update discriminator------------------')
        '''训练判别器'''
        if flag == 0:
            #代表不是谣言
            d1 ,_= self.D(input,len_x)
            loss_d1 = self.criterion(d1,y)

            print(flag,loss_d1)

            input_r = self.G_nr(input,len_x)
            d2,_ = self.D(input_r.detach(),len_x)
            loss_d2 = self.criterion(d2, self.neg_label(y))

            print(flag,loss_d2)

            input_rn = self.G_rn(input_r,len_x)
            d3,_ = self.D(input_rn.detach(),len_x)
            loss_d3 = self.criterion(d3,y)

            loss = loss_d1+loss_d2+loss_d3
            loss.backward(retain_graph=True)
        else:
            #代表是谣言
            d1,_ = self.D(input,len_x)
            loss_d1 = self.criterion(d1, y)

            print(flag,loss_d1)

            input_n = self.G_rn(input, len_x)
            d2,_ = self.D(input_n.detach(),len_x)
            loss_d2 = self.criterion(d2, self.neg_label(y))

            print(flag,loss_d2)

            input_nr = self.G_nr(input,len_x)
            d3,_ = self.D(input_nr.detach(),len_x)
            loss_d3 = self.criterion(d3,y)

            loss = loss_d1 + loss_d2+loss_d3
            loss.backward(retain_graph=True)

        print('update discriminator', loss)
        self.loss_d = loss


    def forward(self, trainloader_n,trainloader_r,test_loader_n,test_loader_r,vis,epochs):
        '''集中训练'''
        for epoch in tqdm.trange(epochs):
            for i,data in enumerate(zip(trainloader_n,trainloader_r)):
                flag = torch.randint(0, 2, [1])

                X, y ,len_x,n_words = data[flag]
                input = self.embeding(X)

                if i % 2 == 0:

                    self.D_optimizer.zero_grad()
                    self.update_D(flag, input, y, len_x)
                    self.D_optimizer.step()

                if i % 5 == 0:
                    if flag == 0:
                        self.G_nr_optimizer.zero_grad()
                        self.update_G_NR(input, y, len_x)
                        self.G_nr_optimizer.step()
                    else:
                        self.G_rn_optimizer.zero_grad()
                        self.update_G_RN(input, y, len_x)
                        self.G_rn_optimizer.step()
                if i % 6 == 0 :
                    tqdm.tqdm.write('epoch : {} loss_grn:{} loss_gnr:{} loss_d:{}'.format(epoch,self.loss_grn,self.loss_gnr,self.loss_d))
                vis.plot('D_loss---globalStep',torch.tensor(self.loss_d))
                vis.plot('loss_grn---globalStep',torch.tensor(self.loss_grn))
                vis.plot('loss_gnr---globalStep',torch.tensor(self.loss_gnr))
            loss_test,acc = self.Testdata(test_loader_n,test_loader_r)
            vis.plot('loss_test',torch.tensor(loss_test))
            vis.plot('acc---epoch',torch.tensor(acc))
            vis.plot('D_loss---epoch',torch.tensor(self.loss_d))
            vis.plot('loss_grn---epoch',torch.tensor(self.loss_grn))
            vis.plot('loss_gnr---epoch',torch.tensor(self.loss_gnr))



    def Testdata(self,test_loader_n,test_loader_r):
        '''run the D model in test dataset'''
        with torch.no_grad():
            out = torch.tensor([[0.0, 0.0]])
            label = torch.tensor([[0.0, 0.0]])
            for valiset in test_loader_n:
                X, y, len_x, n_words = valiset
                input = self.embeding(X)
                out_y, _ = self.D(input, len_x)

                out = torch.cat((out, out_y), 0)
                label = torch.cat((label, y), 0)
            for valiset in test_loader_r:
                X, y, len_x, n_words = valiset
                input = self.embeding(X)
                out_y, _ = self.D(input, len_x)

                out = torch.cat((out, out_y), 0)
                label = torch.cat((label, y), 0)

            loss_test = self.criterion(out, label)
            correct_pred = torch.eq(torch.argmax(out, 1), torch.argmax(label, 1))
            acc = correct_pred.sum().item() / label.size(0)
        return loss_test,acc

    def caculate_cmp(self,Xw,Xe):
        '''计算两个向量相似度'''
        return torch.dist(Xw,Xe,p=1)



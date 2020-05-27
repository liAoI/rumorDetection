from torch import nn
import torch,copy,logging,tqdm,os
import numpy as np
from .base_model import BasicModule
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = ['sans-serif']
plt.rcParams['font.sans-serif'] = ['SimHei']

plt.switch_backend('agg')
import matplotlib.ticker as ticker

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)

PAD=0   #补全
SOS=1   #句子开头
EOS=2   #句子结尾
index2word = np.load('../preprocess/index2word.npy', allow_pickle=True).item()
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

class G_Net(BasicModule):
    '''
            定义生成器
            谣言封装器，为了让谣言像非谣言，让非谣言像谣言
            采用seq2seq模型，即翻译模型
    '''
    def __init__(self,encoder,decoder):
        super(G_Net, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self,seq_in,lengths_seq_in,seq_mask,Test = False):
        seq_out = torch.empty_like(seq_in).to(device)

        batch_size, max_length = seq_in.size()

        if Test: #搜集注意力权重
            atten = torch.zeros(size=(batch_size,max_length,max_length)).to(device)

        outputs = self.encoder(seq_in, seq_mask) #self-attention生成的序列信息
        decoder_input = torch.LongTensor([[SOS for _ in range(batch_size)]]).view(batch_size, 1).to(device)

        decoder_hidden = None
        for t in range(max_length):
            decoder_output, decoder_hidden, atten_weights = self.decoder(decoder_input, decoder_hidden,outputs)
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]]).view(batch_size, 1).to(
                device).detach()
            seq_out[:,t] = decoder_input[:,0]
            if Test:  # 搜集注意力权重
                atten[:,t,:] = atten_weights[:,0,:]

        lengths_seq_in = torch.full_like(lengths_seq_in, max_length,dtype=float).to(device)
        if Test:
            return seq_out,lengths_seq_in,atten
        else:
            return seq_out, lengths_seq_in




class GANMODEL:
    def __init__(self,G_Net,Discriminator,vis,device):
        self.G_rn = G_Net #将谣言伪装为非谣言
        self.G_nr = copy.deepcopy(G_Net)  #将非谣言伪装为谣言
        self.G_rn = self.G_rn.to(device)
        self.G_nr = self.G_nr.to(device)

        self.D = Discriminator.to(device)
        self.G_rn_optimizer = torch.optim.Adam(self.G_rn.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)
        self.G_nr_optimizer = torch.optim.Adam(self.G_nr.parameters(), lr=0.0001,betas=(0.9, 0.98), eps=1e-9)
        self.D_optimizer = torch.optim.Adam(self.D.parameters(), lr=0.0001)

        self.criterion = nn.CrossEntropyLoss().to(device) #损失函数

        self.vis = vis #结果可视化
        self.device = device
        self.lossGrn = 0
        self.lossGnr = 0
        self.lossD = 0

    def train(self,trainloader_n,trainloader_r,vaild_loader_n,vaild_loader_r,epochs):
        '''
        :param trainloader_n: 加载非谣言数据集（训练）
        :param trainloader_r: 加载谣言数据集（训练）
        :param vaild_loader_n: 加载非谣言数据集（验证）
        :param vaild_loader_r: 加载谣言数据集（验证）
        :param epochs:迭代轮数
        :return:None
        '''
        for epoch in tqdm.trange(1,epochs):
            for i,data in enumerate(zip(trainloader_n,trainloader_r)):
                flag = torch.randint(0, 2, [1])

                seq_in,y,lengths_seq_in,seq_mask = data[flag]

                self.update_D(seq_in, y, lengths_seq_in,seq_mask, flag)
                if flag == 0:
                    self.update_Gnr(seq_in, y, lengths_seq_in,seq_mask)
                else:
                    self.update_Grn(seq_in, y, lengths_seq_in,seq_mask)
                # if i % 7 == 0:
                #     self.update_D(seq_in,y,lengths_seq_in,flag)
                # else:
                #     if flag == 0:
                #         self.update_Gnr(seq_in,y,lengths_seq_in)
                #     else:
                #         self.update_Grn(seq_in,y,lengths_seq_in)
                if i % 13 == 0:
                    print('epoch : {} loss_grn:{} loss_gnr:{} loss_d:{}'.\
                                    format(epoch,self.lossGrn,self.lossGnr,self.lossD))



            loss,acc = self.Test(vaild_loader_n,vaild_loader_r)
            self.vis.plot('loss_test', loss)
            self.vis.plot('acc---epoch', acc)
            self.vis.plot('D_loss---epoch', self.lossD.item())
            self.vis.plot('loss_grn---epoch', self.lossGrn.item())
            self.vis.plot('loss_gnr---epoch', self.lossGnr.item())
            self.vis.log('epoch : {} | D_loss : {} | loss_grn : {} | loss_gnr : {} | loss_test : {} | acc : {}'.format(
                epoch,self.lossD.item(),self.lossGrn.item(),self.lossGnr.item(),loss,acc
            ))
            if epoch % 30 == 0:
                self.SaveModel(dir='../',model_name='gan',epoch=epoch,loss=loss,acc=acc)
    def Test(self,loader_n,loader_r):
        loss = 0
        pre = []
        label = []
        count_r = 2 #显示注意力张数
        count_n = 2

        with torch.no_grad():
            for seq_in,y,lengths_seq_in,seq_mask in loader_n:
                #显示生成器结果
                seq_out,lengths_seq_out,atten = self.G_nr(seq_in,lengths_seq_in,seq_mask,True)
                seq_in_n = seq_in[0,:].cpu().numpy().tolist()
                seq_out_r = seq_out[0,:].cpu().numpy().tolist()
                atten = atten[0,:,:].cpu()
                if count_n > 0:
                    self.showAttention(seq_in_n,seq_out_r,atten)
                    count_n -= 1

                out = self.D(seq_in,lengths_seq_in)
                _, topi = out.topk(1)
                pre.extend(topi.view(1,-1).cpu().numpy().tolist()[0])
                label.extend(y.cpu().numpy().tolist())
                loss += self.criterion(out,y)
            for seq_in,y,lengths_seq_in,seq_mask in loader_r:
                # 显示生成器结果
                seq_out, lengths_seq_out, atten = self.G_rn(seq_in, lengths_seq_in,seq_mask, True)
                seq_in_r = seq_in[0, :].cpu().numpy().tolist()
                seq_out_n = seq_out[0, :].cpu().numpy().tolist()
                atten = atten[0, :, :].cpu()
                if count_r > 0:
                    self.showAttention(seq_in_r, seq_out_n, atten)
                    count_r -= 1

                out = self.D(seq_in,lengths_seq_in)
                _, topi = out.topk(1)
                pre.extend(topi.view(1, -1).cpu().numpy().tolist()[0])
                label.extend(y.cpu().numpy().tolist())
                loss += self.criterion(out,y)
        acc = accuracy_score(y_true=label,y_pred=pre)
        return loss,acc

    def showAttention(self, input_sentence, output_words, attentions):

        # Set up figure with colorbar
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy(), cmap='bone')
        fig.colorbar(cax)
        input_sentence = [index2word[word] for word in input_sentence]
        output_words = [index2word[word] for word in output_words]

        # Set up axes
        ax.set_xticklabels(input_sentence
                           , rotation=90)
        ax.set_yticklabels(output_words)

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

        # plt.show()
        self.vis.vis.matplot(plot=plt)

    def neg_label(self, y):
        '''
        :param y: 标签
        :return: 原标签0置为1，1置为0
        '''
        b = torch.ones_like(y).to(self.device)
        result = b + torch.neg(y)

        return result

    def update_Grn(self,seq_in,y,lengths_seq_in,seq_mask):
        ##############################
        #loss：log(D(G(seq_in)))
        #############################
        self.G_rn_optimizer.zero_grad()#梯度清空
        logger.info(msg="更新生成器模型 G = Grn")
        seq_in_rn,lengths_seq_rn = self.G_rn(seq_in,lengths_seq_in,seq_mask) #将谣言伪装为非谣言

        rn_y = self.D(seq_in_rn,lengths_seq_rn)

        loss = self.criterion(rn_y,self.neg_label(y))

        loss.backward() #计算梯度

        self.G_rn_optimizer.step()#更新参数

        self.lossGrn = loss
        self.vis.plot('loss_grn---globalStep', self.lossGrn.item())

    def update_Gnr(self,seq_in,y,lengths_seq_in,seq_mask):
        ##############################
        # loss：log(D(G(seq_in)))
        #############################
        self.G_nr_optimizer.zero_grad()
        logger.info(msg="更新生成器模型 G = Gnr")
        seq_in_nr,lengths_seq_nr = self.G_nr(seq_in,lengths_seq_in,seq_mask) #将非谣言伪装为谣言

        nr_y = self.D(seq_in_nr,lengths_seq_in)

        loss = self.criterion(nr_y,self.neg_label(y))

        loss.backward()

        self.G_nr_optimizer.step()

        self.lossGnr = loss
        self.vis.plot('loss_gnr---globalStep', self.lossGnr.item())

    def update_D(self,seq_in,y,lengths_seq_in,seq_mask,flag):
        ################################
        #loss : log(D(seq_in)) + log(D(G(seq_in)))
        ################################
        logger.info(msg="更新判别器模型 D = D")
        self.D_optimizer.zero_grad()#梯度清零
        if flag == 0:
            #非谣言文本
            out_one = self.D(seq_in,lengths_seq_in)
            loss_one = self.criterion(out_one,y)

            seq_nr,lengths_seq_nr = self.G_nr(seq_in,lengths_seq_in,seq_mask)
            #防止梯度回传到G_nr网络
            out_two = self.D(seq_nr.detach(),lengths_seq_nr.detach())
            loss_two = self.criterion(out_two,y)

            # seq_nrn,lengths_seq_nrn = self.G_rn(seq_nr.detach(),lengths_seq_nr.detach())
            # out_there = self.D(seq_nrn.detach(),lengths_seq_nrn.detach())
            # loss_there = self.criterion(out_there,y)

            loss = loss_one+loss_two#+loss_there

            loss.backward() #计算D的梯度

            self.D_optimizer.step()#更新D的梯度，由于输入的都是seq.detach(),梯度不会传到生成器
            self.lossD = loss
        else:
            #谣言文本
            out_one = self.D(seq_in, lengths_seq_in)
            loss_one = self.criterion(out_one, y)

            seq_rn,lengths_seq_rn = self.G_rn(seq_in.detach(),lengths_seq_in.detach(),seq_mask)
            out_two = self.D(seq_rn,lengths_seq_rn)
            loss_two = self.criterion(out_two,y)

            # seq_rnr,lengths_seq_rnr = self.G_nr(seq_rn.detach(),lengths_seq_rn.detach())
            # out_there = self.D(seq_rnr.detach(),lengths_seq_rnr.detach())
            # loss_there = self.criterion(out_there,y)

            loss = loss_one+loss_two#+loss_there

            loss.backward()

            self.D_optimizer.step()
            self.lossD = loss
        self.vis.plot('D_loss---globalStep', self.lossD.item())

    def SaveModel(self, dir, model_name, epoch,loss,acc):
        directory = os.path.join(dir, model_name,
                            '{}-{}_{}'.format(acc,loss,self.D.n_layers))
        if not os.path.exists(directory):
            os.makedirs(directory)
        torch.save({
                'epoch': epoch,
                'D': self.D.state_dict(),
                'Grn': self.G_rn.state_dict(),
                'Gnr':self.G_nr.state_dict(),
                'D_opt': self.D_optimizer.state_dict(),
                'Grn_opt': self.G_rn_optimizer.state_dict(),
                'Gnr_opt':self.G_nr_optimizer.state_dict(),
        }, os.path.join(directory, '{}_{}.tar'.format(epoch, 'checkpoint')))
    def LoadModel(self,dir):
        checkpoint = torch.load(dir)
        self.D.load_state_dict(checkpoint['D'])
        self.G_rn.load_state_dict(checkpoint['Grn'])
        self.G_nr.load_state_dict(checkpoint['Gnr'])
        self.D_optimizer.load_state_dict(state_dict=checkpoint['D_opt'])
        self.G_rn_optimizer.load_state_dict(state_dict=checkpoint['Grn_opt'])
        self.G_nr_optimizer.load_state_dict(state_dict=checkpoint['Gnr_opt'])
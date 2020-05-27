# coding=utf-8
from Paper.utils.LoadData import IndexOfData
from Paper.utils.visualize import Visualizer
from Paper.models.Transformer import Get_Encoder,Embeddings
from Paper.models.Decoder import AttnDecoder
from Paper.models.GAN import Discriminator
from Paper.models.EncDec import G_Net,GANMODEL

from time import strftime
import logging,copy
import numpy as np
import torch as t
import torch.nn as nn
from torch.utils import data
USE_CUDA = t.cuda.is_available()
device = t.device("cuda" if USE_CUDA else "cpu")


LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)

def my_collate(batch):
    pad  = 0
    batch.sort(key=lambda data: len(data[0]), reverse=True)
    lengths_seq_in = []

    seq_x = []
    y = []
    for (seq_in, seq_out) in batch:
        lengths_seq_in.append(len(seq_in))
        seq_x.append(seq_in)
        y.append(seq_out)
    max_len_x = max(lengths_seq_in)
    y = np.array(y)
    seq_x = np.array([np.pad(s, (0, max_len_x - len(s)), 'constant') for s in seq_x])
    seq_mask = np.zeros_like(seq_x)
    indexNO1 = seq_x > 0
    seq_mask[indexNO1] = 1

    return t.LongTensor(seq_x).to(device=device), \
           t.LongTensor(y).to(device=device), \
           t.tensor(lengths_seq_in, dtype=float).to(device=device),\
           t.tensor(seq_mask,dtype=t.bool).unsqueeze(-2).to(device=device)



if __name__ == '__main__':
    Word2Index = np.load('../preprocess/word2index.npy', allow_pickle=True).item()
    index2word = np.load('../preprocess/index2word.npy', allow_pickle=True).item()
    n_words = np.load('../preprocess/n_words.npy', allow_pickle=True).item()
    word2count = np.load('../preprocess/word2count.npy', allow_pickle=True).item()
    #####################################
    # 定义参数
    #####################################
    dir_rumor = '../datasets/pytorchrumor'
    dir_nonrumor = '../datasets/pytorchunrumor'

    batch_size = 16
    inputSize = 256
    epochs = 100
    atten_mode = ['dot', 'general', 'concat']  # 计算解码器的注意力方式，
    decoder_n_layers = 2  # 解码器层数

    #################################
    # 数据加载、模型定义
    #################################
    train_load_r = IndexOfData(dir=dir_rumor)
    train_iter_r = data.DataLoader(train_load_r, batch_size=batch_size, drop_last=True, collate_fn=my_collate)

    train_load_n = IndexOfData(dir=dir_nonrumor)
    train_iter_n = data.DataLoader(train_load_n, batch_size=batch_size, drop_last=True, collate_fn=my_collate)

    vaild_load_r = IndexOfData(Train=False, dir=dir_rumor)
    vaild_iter_r = data.DataLoader(vaild_load_r, batch_size=batch_size, drop_last=True, collate_fn=my_collate)
    vaild_load_n = IndexOfData(Train=False, dir=dir_nonrumor)
    vaild_iter_n = data.DataLoader(vaild_load_n, batch_size=batch_size, drop_last=True, collate_fn=my_collate)

    VIS = Visualizer(env="GAN", port=1314)
    embedding = Embeddings(vocab=n_words,d_model=inputSize)
    enc = Get_Encoder(src_vocab=n_words,Embeddings=embedding,N=6,d_model=inputSize) #编码器

    dec = AttnDecoder(atten_model=atten_mode[0], hiddenSize=inputSize, embedding=embedding, output_size=n_words,
                      n_layers=decoder_n_layers)

    G = G_Net(encoder=enc, decoder=dec)  # 生成器
    D = Discriminator(hiddenSize=inputSize, embedding=embedding)  # 判别器
##################初始化网络参数
    for p in dec.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    for p in D.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
###########################
    MODEL = GANMODEL(G_Net=G, Discriminator=D, vis=VIS, device=device)  # 模型

    MODEL.train(trainloader_n=train_iter_n, trainloader_r=train_iter_r, \
                vaild_loader_n=vaild_iter_n, vaild_loader_r=vaild_iter_r, epochs=epochs)


from Config import opt
from utils import ShipDataset
import models
from utils import Visualizer
#需要导入以下包

import torch

import numpy as np
import logging
from torch.utils import data


#########配置log日志方便打印#############

LOG_FORMAT = "%(asctime)s -%(filename)s[line:%(lineno)d]- %(name)s - %(levelname)s - %(message)s"
DATE_FORMAT = "%m-%d-%Y %H:%M:%S"

logging.basicConfig(level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)

logger = logging.getLogger(__name__)
######################################

def my_collate(batch):
    '''
        batch里面长度补齐实现策略
        按照句子的平均值来补齐，其中平均值超过2000的，按照2000来截断
    '''
    data_x = [item[0].numpy() for item in batch]
    X = []
    data_y = [item[1].numpy() for item in batch]
    data_len = [x.shape[0] for x in data_x ]

    aver = np.mean(data_len)
    aver = int(aver)  #这里采用强制转换，将平均值转为int型
    pad_col = [aver - x for x in data_len]
    if aver < 2000:
        for cols , data in zip(pad_col,data_x) :
            if cols>0 or cols==0:
                data = np.pad(data,((0,cols),(0,0)),'constant')
                X.append(data)
            else:
                X.append(data[0:aver,:])
    else:
        for len, data in zip(data_len, data_x):
            if len<2000:
                data = np.pad(data, ((0, 2000-len),(0,0)), 'constant')
                X.append(data)
            else:
                X.append(data[0:2000, :])
    # logger.info('---结束打包了，宝贝！---')
    X= torch.tensor(X)
    Y = torch.tensor(data_y)
    return X,Y
    # return batch
#测试集上得出精度和损失值
def Test_on_data(valid_loader,model,criterion):
    out = torch.tensor([[0.0, 0.0]]).cuda(device=0)
    y = torch.tensor([[0.0, 0.0]]).cuda(device=0)
    for valiset in valid_loader:
        v_X, v_Y = valiset
        v_X = v_X.cuda(device=0)
        v_Y = v_Y.cuda(device=0)
        with torch.no_grad():
            out_y, _ = model(v_X)

        out = torch.cat((out, out_y), 0)
        y = torch.cat((y, v_Y), 0)

    loss_test = criterion(out, y)
    correct_pred = torch.eq(torch.argmax(out, 1), torch.argmax(y, 1))
    acc = correct_pred.sum().item() / y.size(0)
    return loss_test,acc

#加载训练数据
train_data = ShipDataset(vali=4)
train_loader = data.DataLoader(train_data, batch_size=5, num_workers=2, shuffle=False,collate_fn=my_collate)
#加载测试数据
valid_data = ShipDataset(vali=4,Train=False)
valid_loader = data.DataLoader(valid_data, batch_size=2, num_workers=2, shuffle=False,collate_fn=my_collate)

#训练开始，首先是加载模型
# model = models.RNNModel(opt.rnn_inputsize,opt.rnn_hidden_size,opt.rnn_layers).cuda(device=0)
# print(model)
model = models.rnn(opt.rnn_inputsize,opt.rnn_hidden_size,opt.rnn_layers).cuda(device=0)
print(model)

#画图
vis = Visualizer(env = 'nlp')

# vis.plot('loss',2.9)
# vis.plot('loss',3.8)
#定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=opt.rnn_lr,weight_decay=0.0005)

#定义二分类损失函数
criterion = torch.nn.BCELoss().cuda(device=0)

#动态调整学习率
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3, verbose=True,
                                               threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)

for epoch in range(opt.rnn_epochs):
    for i, trainset in enumerate(train_loader):
        X, Y = trainset

        X = X.cuda(device=0)
        Y = Y.cuda(device=0)
        out_y,_ = model(X)

        #计算loss值
        loss = criterion(out_y, Y)

        #更新权重和偏向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        #每更新一百次打印一下loss，学习率
        if i % 100 == 0:
            print('batch_loss: {0},学习率为{1}'.format(loss, optimizer.defaults['lr']))
    # loss不下降就更新学习率
    scheduler.step(loss)

    loss_test,acc=Test_on_data(valid_loader, model, criterion)
    vis.plot('loss',loss_test.item())
    vis.plot('acc',acc)
    print('第 {0} 轮训练精度为 {1} 损失值为{2}'.format(epoch + 1, acc,loss_test))

import torch.nn as nn
import torch as t
import time
class BasicModule(nn.Module):
    '''
    封装了nn.Module,主要提供save和load两个方法
    其他的模型必须继承此类
    '''
    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self))   #模型的默认名字

    def load(self,path):
        '''
        :param path: 模型加载路径
        :return: None
        '''
        self.load_state_dict(t.load(path))

    def save(self,name=None):
        '''

        :param name: 保存的模型名字，默认使用“模型名字+时间”作为文件名
        :return: name 保存的模型文件名
        '''
        if name is None:
            prefix = 'checkpoints/'+self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')

        t.save(self.state_dict(),name)
        return name
import warnings
class DefaultConfig(object):

    '''
    这里定义基本配置
    '''
    model = 'rnn'  #使用的模型，名字必须与models/__init__.py中名字一致

    # RNN训练的参数设置
    rnn_hidden_size = 100
    rnn_inputsize = 20  # 这是根据word2vec训练出来的词向量size决定的
    rnn_layers = 2
    rnn_epochs = 100
    rnn_lr = 1e-3

def parse(self, kwargs):
        '''
        根据字典kwargs更新config参数
        :param kwargs: 字典参数
        :return: None
        '''
        # 更新配置参数
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: config has not attribute %s" % k)
            setattr(self, k, v)

        # 打印配置信息
        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, getattr(self, k))

DefaultConfig.parse = parse
opt = DefaultConfig()
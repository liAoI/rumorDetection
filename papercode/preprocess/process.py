import numpy as np
import os
import json,re
from torch.utils import data
from tqdm import tqdm


def CleanStr(s):
    s = re.sub(r'([-【】！、。，？“”()（）.!?])', r'', s)  #
    s = s.split(' ')  # 按照空格将词分开
    s = list(filter(None, s))  # 去掉空字符
    return s  # 返回的是list
if __name__ == '__main__':
    '''
        构建词袋，wordToindex,indexToword
    '''
    dir = '../datasets'
    flist = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if os.path.splitext(file)[1] == '.json':
                flist.append(os.path.join(root, file))

    Word2Index = {'SOS': 1, 'EOS': 2}
    index2word = {1: 'SOS', 2: 'EOS'}
    n_words = 2
    word2count = {}

    tqdm.write('------正在构建词典------')
    for filename in tqdm(flist):
        with open(filename, 'r', encoding='utf-8') as f:
            fjson = json.load(f)
            s = fjson['text']
            s = CleanStr(s)
            for word in s:
                if word not in Word2Index:
                    n_words += 1
                    Word2Index[word] = n_words
                    index2word[n_words] = word
                    word2count[word] = 1
                else:
                    word2count[word] += 1
    tqdm.write('------词典索引构建成功------')
    np.save('word2index.npy', Word2Index)
    np.save('index2word.npy', index2word)
    np.save('n_words.npy', n_words)
    np.save('word2count.npy',word2count)



# rumorDetection

这只是自己写的一些实验的代码。用RNN(lstm,gru),cnn+lstm做的谣言检测。实现的论文主要以ma Jing的为主，也有一些其他的论文。

20190828 --更新-- 通过torch.gather，torch.nn.utils.rnn.pad_packed_sequence，torch.nn.utils.rnn.pack_padded_sequence实现数据批次补齐后出现的一些导致结果偏差的问题。

#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 15:20
@Author  : Xie Cheng
@File    : rnn.py
@Software: PyCharm
@desc: 循环神经网络结构
"""
from torch import nn


class Rnn(nn.Module):
    def __init__(self, input_num=1, hidden_num=32, layer_num=1, output_num=1, seq_len=1000):
        super(Rnn, self).__init__()
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.seq_len = seq_len  # 序列长度

        self.rnn = nn.RNN(
            input_size=input_num,
            hidden_size=hidden_num,
            num_layers=layer_num,
            nonlinearity='relu',
            batch_first=True  # 输入(batch, seq, feature)
        )

        self.Out = nn.Linear(hidden_num, output_num)

    def forward(self, u, h_state):
        """
        :param u: input输入
        :param h_state: 循环神经网络状态量
        :return:
        """
        r_out, h_state_next = self.rnn(u, h_state)
        r_out_reshaped = r_out.view(-1, self.hidden_num)  # to 2D data
        outs = self.Out(r_out_reshaped)
        outs = outs.view(-1, self.seq_len, self.output_num)  # to 3D data

        return outs, h_state_next

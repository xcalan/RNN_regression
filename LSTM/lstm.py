#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 22:19
@Author  : Xie Cheng
@File    : lstm.py
@Software: PyCharm
@desc: LSTM网络结构
"""
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_num=1, hidden_num=32, layer_num=1, output_num=1, seq_len=1000):
        super(LSTM, self).__init__()
        self.hidden_num = hidden_num
        self.output_num = output_num
        self.layer_num = layer_num
        self.seq_len = seq_len  # 序列长度

        self.lstm = nn.LSTM(
            input_size=input_num,
            hidden_size=hidden_num,
            num_layers=layer_num,
            batch_first=True  # 输入(batch, seq, feature)
        )

        self.Out = nn.Linear(hidden_num, output_num)

    def forward(self, u, h_0, c_0):
        """
        :param u: input输入
        :param h_0, c_0: 循环神经网络状态量
        :return:
        """
        r_out, (h_n, c_n) = self.lstm(u, (h_0, c_0))
        r_out_reshaped = r_out.view(-1, self.hidden_num)  # to 2D data
        outs = self.Out(r_out_reshaped)
        outs = outs.view(-1, self.seq_len, self.output_num)  # to 3D data

        return outs, (h_n, c_n)

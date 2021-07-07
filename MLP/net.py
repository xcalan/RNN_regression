#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 21:48
@Author  : Xie Cheng
@File    : net.py
@Software: PyCharm
@desc: BP神经网络
"""
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, input_num=1, hidden_num=32, output_num=1):
        super(Net, self).__init__()
        self.Hidden = nn.Linear(input_num, hidden_num)
        self.Out = nn.Linear(hidden_num, output_num)
        self.Relu = nn.ReLU()

    def forward(self, x):
        """
        :param x: input输入
        :return:
        """
        x = self.Hidden(x.view([-1, 1]))
        x = self.Relu(x)
        x = self.Out(x)
        x = x.squeeze()

        return x


# if __name__ == '__main__':
#     net = Net()
#     x = torch.randn([10])
#     print(net(x))

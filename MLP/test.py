#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 22:06
@Author  : Xie Cheng
@File    : test.py
@Software: PyCharm
@desc: 网络测试
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from myfunction import MyDataset

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print('You are using: ' + str(device))

# batch size
batch_size_test = 200

# 导入数据
filename = '../data/data.csv'
dataset_test = MyDataset(filename)
test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

criterion = nn.MSELoss()  # mean square error


# rnn 测试
def test():
    net_test = torch.load('..\\model\\mlp_model.pkl')  # load model
    test_loss = 0
    net_test.eval()
    with torch.no_grad():
        for idx, (sin_input, cos_output) in enumerate(test_loader):
            prediction = net_test(sin_input.to(device))
            if idx == 0:
                predict_value = prediction.squeeze()
                real_value = cos_output.squeeze()
            else:
                predict_value = torch.cat([predict_value, prediction.squeeze()], dim=0)
                real_value = torch.cat([real_value, cos_output.squeeze()], dim=0)

            loss = criterion(prediction, cos_output.to(device))
            test_loss += loss.item()

    print('Test set: Avg. loss: {:.9f}'.format(test_loss))
    return predict_value, real_value


if __name__ == '__main__':
    # 模型测试
    print("testing...")
    p_v, r_v = test()

    # 对比图
    plt.plot(p_v.cpu(), c='green')
    plt.plot(r_v.cpu(), c='orange', linestyle='--')
    plt.show()
    print("stop testing!")

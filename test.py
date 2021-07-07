#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 16:15
@Author  : Xie Cheng
@File    : test.py
@Software: PyCharm
@desc: 测试过程
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
batch_size_test = 20

# 导入数据
filename = './data/data.csv'
dataset_test = MyDataset(filename)
test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, drop_last=True)

criterion = nn.MSELoss()  # mean square error


# rnn 测试
def test_rnn():
    net_test = torch.load('.\\model\\rnn_model.pkl')  # load model
    hidden_state = None
    test_loss = 0
    net_test.eval()
    with torch.no_grad():
        for idx, (sin_input, cos_output) in enumerate(test_loader):
            sin_input_np = sin_input.numpy()  # 1D
            cos_output_np = cos_output.numpy()  # 1D

            sin_input_torch = torch.from_numpy(sin_input_np[np.newaxis, :, np.newaxis])  # 3D
            cos_output_torch = torch.from_numpy(cos_output_np[np.newaxis, :, np.newaxis])  # 3D

            prediction, hidden_state = net_test(sin_input_torch.to(device), hidden_state)

            if idx == 0:
                predict_value = prediction.squeeze()
                real_value = cos_output_torch.squeeze()
            else:
                predict_value = torch.cat([predict_value, prediction.squeeze()], dim=0)
                real_value = torch.cat([real_value, cos_output_torch.squeeze()], dim=0)

            loss = criterion(prediction, cos_output_torch.to(device))
            test_loss += loss.item()

    print('Test set: Avg. loss: {:.9f}'.format(test_loss))
    return predict_value, real_value


if __name__ == '__main__':
    # 模型测试
    print("testing...")
    p_v, r_v = test_rnn()

    # 对比图
    plt.plot(p_v.cpu(), c='green')
    plt.plot(r_v.cpu(), c='orange', linestyle='--')
    plt.show()
    print("stop testing!")

#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 15:14
@Author  : Xie Cheng
@File    : train.py
@Software: PyCharm
@desc: 训练过程
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

from myfunction import MyDataset
from RNN.rnn import Rnn

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
batch_size_train = 20
# total epoch(总共训练多少轮)
total_epoch = 1000

# 1. 导入训练数据
filename = '../data/data.csv'
dataset_train = MyDataset(filename)
train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False, drop_last=True)

# 2. 构建模型，优化器
rnn = Rnn(seq_len=batch_size_train).to(device)
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.001, momentum=0.8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # mean square error
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 1  # 网络训练过程中最大的loss


# 3. 模型训练
def train_rnn(epoch):
    hidden_state = None  # 隐藏状态初始化
    global total_loss
    mode = True
    rnn.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和

    for idx, (sin_input, cos_output) in enumerate(train_loader):
        sin_input_np = sin_input.numpy()  # 1D
        cos_output_np = cos_output.numpy()  # 1D

        sin_input_torch = Variable(torch.from_numpy(sin_input_np[np.newaxis, :, np.newaxis]))  # 3D
        cos_output_torch = Variable(torch.from_numpy(cos_output_np[np.newaxis, :, np.newaxis]))  # 3D

        prediction, hidden_state = rnn(sin_input_torch.to(device), hidden_state)

        # 要把 h_state 重新包装一下才能放入下一个 iteration, 不然会报错!!!
        hidden_state = Variable(hidden_state.data).to(device)

        loss = criterion(prediction, cos_output_torch.to(device))  # cross entropy loss
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        if idx == len(train_loader) - 1:
            print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                torch.save(rnn, '..\\model\\rnn_model.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练1000轮
        train_rnn(i)
    print("Stop Training!")

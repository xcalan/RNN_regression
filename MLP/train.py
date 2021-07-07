#!/usr/bin/env python3
# encoding: utf-8
"""
@Time    : 2021/7/7 21:56
@Author  : Xie Cheng
@File    : train.py
@Software: PyCharm
@desc: 网络训练
"""
import torch
from torch import nn
from torch.utils.data import DataLoader
import numpy as np
from torch.autograd import Variable

from myfunction import MyDataset
from MLP.net import Net

# device GPU or CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('You are using: ' + str(device))

# batch size
batch_size_train = 200
# total epoch(总共训练多少轮)
total_epoch = 1000

# 1. 导入训练数据
filename = '../data/data.csv'
dataset_train = MyDataset(filename)
train_loader = DataLoader(dataset_train, batch_size=batch_size_train, shuffle=False, drop_last=True)

# 2. 构建模型，优化器
net = Net().to(device)
optimizer = torch.optim.SGD(net.parameters(), lr=0.05, momentum=0.8)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)  # Learning Rate Decay
criterion = nn.MSELoss()  # mean square error
train_loss_list = []  # 每次epoch的loss保存起来
total_loss = 100  # 网络训练过程中最大的loss


# 3. 模型训练
def train(epoch):
    global total_loss
    mode = True
    net.train(mode=mode)  # 模型设置为训练模式
    loss_epoch = 0  # 一次epoch的loss总和

    for idx, (sin_input, cos_output) in enumerate(train_loader):
        sin_input_torch = Variable(sin_input)
        prediction = net(sin_input_torch.to(device))

        loss = criterion(prediction, cos_output.to(device))  # MSE
        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # back propagation, compute gradients
        optimizer.step()  # apply gradients

        loss_epoch += loss.item()  # 将每个batch的loss累加，直到所有数据都计算完毕
        if idx == len(train_loader) - 1:
            print('Train Epoch:{}\tLoss:{:.9f}'.format(epoch, loss_epoch))
            train_loss_list.append(loss_epoch)
            if loss_epoch < total_loss:
                total_loss = loss_epoch
                torch.save(net, '..\\model\\mlp_model.pkl')  # save model


if __name__ == '__main__':
    # 模型训练
    print("Start Training...")
    for i in range(total_epoch):  # 模型训练1000轮
        train(i)
    print("Stop Training!")

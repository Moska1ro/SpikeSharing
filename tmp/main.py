from spikevgg import VGG
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import Data
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class ConfusionMatrix(object):
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.mat = None

    def update(self, a, b):
        n = self.num_classes
        if self.mat is None:
            # 创建混淆矩阵
            self.mat = torch.zeros((n, n), dtype=torch.int64, device=a.device)
        with torch.no_grad():
            # 寻找GT中为目标的像素索引
            k = (a >= 0) & (a < n)
            # 统计像素真实类别a[k]被预测成类别b[k]的个数(这里的做法很巧妙)
            inds = n * a[k].to(torch.int64) + b[k]
            self.mat += torch.bincount(inds, minlength=n ** 2).reshape(n, n)

    def reset(self):
        if self.mat is not None:
            self.mat.zero_()

    def get_acc(self):
        h = self.mat.float()
        acc_global = torch.diag(h).sum() / h.sum()
        return acc_global

    def compute(self):
        h = self.mat.float()
        # 计算全局预测准确率(混淆矩阵的对角线为预测正确的个数)
        acc_global = torch.diag(h).sum() / h.sum()
        # 计算每个类别的准确率
        acc = torch.diag(h) / h.sum(1)
        # 计算每个类别预测与真实目标的iou
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h))
        recall = torch.diag(h) / h.sum(0)
        f1 = 2 * acc * recall / (acc + recall)
        return acc_global, acc, iu, f1


if __name__ == '__main__':
    model = VGG('vgg16', T=8)
    # print(model)
    T = 8
    d = Data(128, './cifar10')
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.025, momentum=0.9, weight_decay=5e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[70, 100, 125], gamma=0.1)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 250)
    mat = ConfusionMatrix(10)
    for epoch in range(250):
        model.train()
        losses = []
        for inputs, labels in d.trainLoader:
            inputs = inputs.repeat(T, 1, 1, 1, 1)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            # print(outputs.shape)
            optimizer.zero_grad()
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            # scheduler.step()
            functional.reset_net(model)
            losses.append(loss.item())
        print(f'epoch{epoch}:{np.mean(losses)}')
        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                for inputs, labels in d.testLoader:
                    inputs = inputs.repeat(T, 1, 1, 1, 1)
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    mat.update(labels.flatten(), outputs.argmax(1).flatten())
                    functional.reset_net(model)
                acc = mat.get_acc()
                print(acc.item())
                mat.reset()
            torch.save(model.state_dict(), 'save.pt')

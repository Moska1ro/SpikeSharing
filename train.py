from spikevgg import VGG
from spiking_resnet import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import Data
import sys
from spikingjelly.activation_based import neuron, functional, surrogate, layer

if __name__ == '__main__':
    sys.stdout = open('./resnet_18_23_13.txt', 'w')
    honey = 27 * [10]
    print(honey)
    model = resnet('resnet18')
    # print(model)
    T = 20
    batch_size = 128
    d = Data(batch_size=batch_size, data_path='./cifar10')
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    model.to(device)
    model.train()
    loss_func = nn.CrossEntropyLoss().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    optimizer = optim.Adam(model.parameters(),lr=0.001)
    best_acc = 0
    for epoch in range(256):
        model.train()
        losses = []
        for inputs, labels in d.trainLoader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = 0.
            for t in range(0, T):
                outputs += model(inputs)
            outputs = outputs / T
            loss = loss_func(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            functional.reset_net(model)
            losses.append(loss.item())
        print(f'epoch{epoch}:{np.mean(losses)}')
        if (epoch + 1) % 5 == 0:
            model.eval()
            total_test_loss = 0
            total_accuracy = 0
            with torch.no_grad():
                for inputs, labels in d.testLoader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = 0.
                    for t in range(0, T):
                        outputs += model(inputs)
                    outputs = outputs / T
                    # mat.update(labels.flatten(), outputs.argmax(1).flatten())
                    loss_e = loss_func(outputs, labels)
                    total_test_loss += loss_e.item()
                    accuracy = (outputs.argmax(1) == labels).sum()
                    total_accuracy = total_accuracy + accuracy
                    functional.reset_net(model)
                # acc = mat.get_acc()
                # print(acc.item())
                # mat.reset()
                print("test set Loss: {}".format(total_test_loss))
                print("test set accuracy: {}".format(total_accuracy / d.testlen))
                if total_accuracy / d.testlen > best_acc:
                    best_acc = total_accuracy / d.testlen
                    torch.save(model.state_dict(), 'save_23_13_T=20.pt')
    print('best acc:', best_acc)
    sys.stdout = sys.__stdout__

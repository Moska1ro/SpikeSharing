from spiking_resnet import resnet
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from data import Data
import sys
from spikingjelly.activation_based import neuron, functional, surrogate, layer


if __name__ == '__main__':
    T = 20
    batch_size = 256
    d = Data(batch_size=batch_size, data_path='./cifar10')
    device = torch.device("cuda") if torch.cuda.is_available() else 'cpu'
    loss_func = nn.CrossEntropyLoss().to(device)

    model = resnet('resnet18')
    ckpt = torch.load('./pts/save_23_54_T=20.pt')
    model.load_state_dict(ckpt)
    model.to(device)

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
            loss_e = loss_func(outputs, labels)
            total_test_loss += loss_e.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy = total_accuracy + accuracy
            functional.reset_net(model)
        print("test set Loss: {}".format(total_test_loss))
        print("test set accuracy: {}".format(total_accuracy / d.testlen))
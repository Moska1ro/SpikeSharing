import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.activation_based import neuron, functional, surrogate, layer

conv_num_cfg = {
    'resnet18': 8,
    'resnet34': 16,
    'resnet50': 16,
    'resnet101': 33,
    'resnet152': 50
}


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, honey, index, stride=1):
        super(BasicBlock, self).__init__()
        self.index = index
        self.conv1 = layer.Conv2d(in_planes, int(planes * honey[self.index] / 10), kernel_size=3, stride=stride, padding=1,
                               bias=False)
        self.bn1 = layer.BatchNorm2d(int(planes * honey[self.index] / 10))
        self.conv2 = layer.Conv2d(int(planes * honey[self.index] / 10), planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = layer.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                layer.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                layer.BatchNorm2d(self.expansion * planes)
            )
        self.neuron1 = neuron.IFNode(surrogate_function=surrogate.ATan())
        self.neuron2 = neuron.IFNode(surrogate_function=surrogate.ATan())

    def forward(self, x):
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = self.neuron2(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, honey=None):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.honey = honey
        self.current_conv = 0

        self.conv1 = layer.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = layer.BatchNorm2d(64)
        self.maxpool = layer.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layers(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layers(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layers(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layers(block, 512, num_blocks[3], stride=2)

        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.neuron1 = neuron.IFNode(surrogate_function=surrogate.ATan())

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layers(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes,
                                self.honey, self.current_conv, stride))
            self.current_conv += 1
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.neuron1(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def resnet(cfg, honey=None, num_classes=10):
    if honey is None:
        honey = conv_num_cfg[cfg] * [10]
    if cfg == 'resnet18':
        return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, honey=honey)
    elif cfg == 'resnet34':
        return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, honey=honey)


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


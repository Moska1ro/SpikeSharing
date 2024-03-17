import torch.nn as nn
from spikingjelly.activation_based import neuron, functional, surrogate, layer

cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes=10, T=4):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        # self.classifier = layer.Linear(512, num_classes)
        self.classifier = nn.Sequential(
            layer.Linear(512 * 7 * 7, 4096),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(),
            layer.Linear(4096, 4096),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(),
            layer.Linear(4096, 10),
        )
        self.T = T
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        # x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.classifier(out)
        out = out.mean(0)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [layer.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           layer.BatchNorm2d(x),
                           neuron.IFNode(surrogate_function=surrogate.ATan())]
                in_channels = x
        layers += [layer.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


# honeysource: 1d向量，值限制在0-9
class BeeVGG(nn.Module):
    def __init__(self, vgg_name, honeysource, T):
        super(BeeVGG, self).__init__()
        self.honeysource = honeysource
        self.features = self._make_layers(cfg[vgg_name])
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        # self.classifier = layer.Linear(int(512 * honeysource[len(honeysource) - 1] / 10), 10)
        self.classifier = nn.Sequential(
            layer.Linear(512 * 7 * 7, 4096),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(),
            layer.Linear(4096, int(512 * honeysource[len(honeysource) - 1] / 10)),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.Dropout(),
            layer.Linear(int(512 * honeysource[len(honeysource) - 1] / 10), 10),
        )
        self.T = T
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
       #  x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        out = self.features(x)
        out = self.avgpool(out)
        out = out.view(out.size(0), out.size(1), -1)
        out = self.classifier(out)
        out = out.mean(0)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        index = 0
        Mlayers = 0
        for x_index, x in enumerate(cfg):
            if x == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
                Mlayers += 1
            else:
                x = int(x * self.honeysource[x_index - Mlayers] / 10)
                if x == 0:
                    x = 1
                layers += [layer.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           layer.BatchNorm2d(x),
                           neuron.IFNode(surrogate_function=surrogate.ATan())]
                in_channels = x
        layers += [layer.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


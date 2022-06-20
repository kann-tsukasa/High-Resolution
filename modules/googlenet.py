'''GoogLeNet with PyTorch.'''

from torchvision.models import googlenet
import torch
import torch.nn as nn


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_layers = googlenet()
        self.linear = nn.Linear(1000, 12)

    def forward(self, x):
        try:
            out, aux2, aux1 = self.pre_layers(x)
        except:
            out = self.pre_layers(x)
        out = self.linear(out)
        return out


def test():
    net = GoogLeNet()
    x = torch.randn(1,3,192,192)
    y = net(x)
    print(y.size())


if __name__ == '__main__':
    test()

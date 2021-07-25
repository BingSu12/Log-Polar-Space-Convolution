import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from LogPoolingCovDis import LogPoolingCovLayer


__all__ = [
    'VGG_lpc', 'vgg11_lpc', 'vgg11_bn_lpc', 'vgg13_lpc', 'vgg13_bn_lpc', 'vgg16_lpc', 'vgg16_bn_lpc',
    'vgg19_bn_lpc', 'vgg19_lpc',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG_lpc(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG_lpc, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

        self.pre = nn.Sequential(
            LogPoolingCovLayer(5, 5, stride=1, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=3),
            nn.Conv2d(3, 64, kernel_size=(4,3), stride=(4,3), padding=0),
        )
        self.centerconv = nn.Conv2d(3, 64, kernel_size=1, padding=0)
        self.post = nn.Sequential(
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.pre(x) + self.centerconv(x)
        # print(x.shape)
        x = self.post(y)

        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 64 #3
    count = 0
    for v in cfg:
        if v == 'M':
            count = count + 1
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            if count<0:
                logpl = LogPoolingCovLayer(9, 9, stride=1, pool_type='avg_pool', num_levels=2, ang_levels=8, facbase=3)
                conv2d = nn.Conv2d(in_channels, v, kernel_size=(4,4), stride=(4,4), padding=0)
                
                if batch_norm:
                    layers += [logpl, conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [logpl, conv2d, nn.ReLU(inplace=True)]
                in_channels = v
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'E_p': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11_lpc(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_lpc(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn_lpc(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG_lpc(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13_lpc(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_lpc(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn_lpc(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG_lpc(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16_lpc(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_lpc(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn_lpc(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG_lpc(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19_lpc(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG_lpc(make_layers(cfg['E_p']), **kwargs)
    return model


def vgg19_bn_lpc(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG_lpc(make_layers(cfg['E_p'], batch_norm=True), **kwargs)
    return model

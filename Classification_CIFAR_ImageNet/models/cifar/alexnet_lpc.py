import torch.nn as nn
from LogPoolingCovDis import LogPoolingCovLayer

__all__ = ['alexnet_lpc']



class AlexNet(nn.Module):

    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.logp1 = nn.Sequential(
            LogPoolingCovLayer(11, 11, stride=4, pool_type='avg_pool', num_levels=3, ang_levels=8, facbase=2),
            nn.Conv2d(3, 64, kernel_size=(6,4), stride=(6,4), padding=0),
        )
        self.centerconv1 = nn.Conv2d(3, 64, kernel_size=1, stride=4, padding=0)
        self.lowfeatures = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.logp2 = LogPoolingCovLayer(9, 9, stride=1, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=3)
        self.logconv2 = nn.Conv2d(64, 192, kernel_size=(4,3), stride=(4,3), padding=0)
        self.centerconv2 = nn.Conv2d(64, 192, kernel_size=1, stride=1, padding=0)
            
        self.highfeatures = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        y3 = self.logp1(x) + self.centerconv1(x)
        output = self.lowfeatures(y3)
        y1 = self.logconv2(self.logp2(output))
        y2 = self.centerconv2(output)
        z = y1+y2
        x = self.highfeatures(z)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def alexnet_lpc(**kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    """
    model = AlexNet(**kwargs)
    return model

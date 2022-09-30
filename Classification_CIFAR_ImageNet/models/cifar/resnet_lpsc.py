from __future__ import absolute_import

import torch
import torch.nn as nn
import math
from LogPoolingCovDis import LogPoolingCovLayer

__all__ = ['resnet_lpsc']

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlockLogS(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockLogS, self).__init__()
        self.logpl1 = LogPoolingCovLayer(9, 9, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x1 = self.logpl1(x)
        out = self.conv1(x1) + self.centerconv1(x)
        out = self.bn1(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckLogS(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckLogS, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.logpl = LogPoolingCovLayer(5, 5, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out2 = self.logpl(out)
        out = self.conv2(out2) + self.centerconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class BasicBlockLog(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockLog, self).__init__()
        self.logpl1 = LogPoolingCovLayer(5, 5, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=2)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.logpl2 = LogPoolingCovLayer(5, 5, stride=1, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        x1 = self.logpl1(x)
        out = self.conv1(x1) + self.centerconv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out2 = self.logpl2(out)
        out = self.conv2(out2) + self.centerconv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckLog(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckLog, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.logpl = LogPoolingCovLayer(5, 5, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=2)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out2 = self.logpl(out)  #
        out = self.conv2(out2) + self.centerconv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class BasicBlockHL(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlockHL, self).__init__()
        self.conv1 = conv3x3(inplanes, planes//2, stride)
        self.logpl = LogPoolingCovLayer(5, 5, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=3)
        self.lpsc1 = nn.Conv2d(inplanes, planes//2, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv2 = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=stride, padding=0)
        
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out1 = self.conv1(x)
        out2 = self.lpsc1(self.logpl(x)) + self.centerconv2(x)
        out = torch.cat((out1,out2),1)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BottleneckHL(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BottleneckHL, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes//2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.logpl = LogPoolingCovLayer(5, 5, stride=stride, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=3)
        self.lpsc1 = nn.Conv2d(planes, planes//2, kernel_size=(4,3), stride=(4,3), padding=0, bias=False)
        self.centerconv2 = nn.Conv2d(planes, planes//2, kernel_size=1, stride=stride, padding=0)

        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out1 = self.conv2(out)
        out2 = self.lpsc1(self.logpl(out)) + self.centerconv2(out)
        out = torch.cat((out1,out2),1)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out





class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out







class ResNet(nn.Module):

    def __init__(self, depth, num_classes=1000, block_name='BasicBlock'):
        super(ResNet, self).__init__()
        # Model type specifies number of layers for CIFAR-10 model
        if block_name.lower() == 'basicblock':
            assert (depth - 2) % 6 == 0, 'When use basicblock, depth should be 6n+2, e.g. 20, 32, 44, 56, 110, 1202'
            n = (depth - 2) // 6
            block = BasicBlock
            blockhl = BasicBlockHL
        elif block_name.lower() == 'bottleneck':
            assert (depth - 2) % 9 == 0, 'When use bottleneck, depth should be 9n+2, e.g. 20, 29, 47, 56, 110, 1199'
            n = (depth - 2) // 9
            block = Bottleneck
            blockhl = BottleneckHL
        else:
            raise ValueError('block_name shoule be Basicblock or Bottleneck')


        self.inplanes = 16
        self.conv1 = conv3x3(3, 8, stride=1)
        self.logpl = LogPoolingCovLayer(5, 5, stride=1, pool_type='avg_pool', num_levels=2, ang_levels=6, facbase=3)
        self.lpsc1 = nn.Conv2d(3, 8, kernel_size=(4,3), stride=(4,3), padding=0)
        self.centerconv1 = nn.Conv2d(3, 8, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(blockhl, 16, n)
        self.layer2 = self._make_layer(block, 32, n, stride=2)
        self.layer3 = self._make_layer(block, 64, n, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.lpsc1(self.logpl(x)) + self.centerconv1(x)
        x = torch.cat((x1,x2),1)

        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_lpsc(**kwargs):
    """
    Constructs a ResNet model.
    """
    return ResNet(**kwargs)

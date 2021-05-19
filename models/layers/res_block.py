from torch import nn

__all__ = ['BasicBlock', 'BottleNeck']


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, activation_layer=nn.ReLU,
                 batch_norm=True, downsampling=False):
        super(BasicBlock, self).__init__()
        self.downsampling = downsampling
        self.batch_norm = batch_norm

        if self.downsampling:
            planes = in_planes * 2
            self.conv1 = conv3x3(in_planes, planes, stride=2)

            self.skip = conv1x1(in_planes, planes, stride=2)
            self.skip_bn = nn.BatchNorm2d(planes)
        else:
            planes = in_planes
            self.conv1 = conv3x3(in_planes, planes)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(planes)
        self.act1 = activation_layer()
        self.conv2 = conv3x3(planes, planes)
        if batch_norm:
            self.bn2 = nn.BatchNorm2d(planes)

        self.act2 = activation_layer()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.batch_norm:
            out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        if self.batch_norm:
            out = self.bn2(out)

        if self.downsampling:
            identity = self.skip(identity)
            identity = self.skip_bn(identity)

        out += identity
        out = self.act2(out)
        return out


class BottleNeck(nn.Module):
    def __init__(self):
        super(BottleNeck, self).__init__()
        raise NotImplementedError


def conv3x3(in_planes: int, out_planes: int, stride: int = 1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

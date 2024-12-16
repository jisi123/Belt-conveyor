import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

#定义一个名为 BasicBlock 的类，它继承自 nn.Module。
class BasicBlock(nn.Module):
    expansion = 1#定义一个类变量 expansion，它表示每个残差块的输出通道数与输入通道数的比例。
#定义两个卷积层。定义两个批量归一化层。定义一个 ReLU 激活函数。定义下采样层和步长。
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
#定义前向传播函数，保存输入 x 作为残差
    def forward(self, x):
        residual = x
#应用第一个卷积层。应用批量归一化和 ReLU 激活。应用第二个卷积层。再次应用批量归一化和 ReLU 激活。
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
#如果存在下采样层，则应用它。
        if self.downsample is not None:
            residual = self.downsample(x)
#将残差添加到输出中。应用 ReLU 激活。
        out += residual
        out = self.relu(out)

        return out

#定义一个名为 Bottleneck 的类，它也继承自 nn.Module。
class Bottleneck(nn.Module):
    expansion = 4#定义一个类变量 expansion，表示每个瓶颈块的输出通道数与输入通道数的比例。

#构造函数，用于初始化 Bottleneck。这里有三个卷积层。
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
#定义前向传播函数。
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


class B2_ResNet(nn.Module):
    # ResNet50 with two branches定义一个名为 B2_ResNet 的类，它继承自 nn.Module。
    def __init__(self):
        # self.inplanes = 128
        self.inplanes = 64#初始化输入通道数
        super(B2_ResNet, self).__init__()
#卷积，定义多个残差块层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3_1 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_1 = self._make_layer(Bottleneck, 512, 3, stride=2)

        self.inplanes = 512
        self.layer3_2 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4_2 = self._make_layer(Bottleneck, 512, 3, stride=2)
#遍历模型的所有模块并进行权重初始化。
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

   # 定义一个辅助函数，用于创建多个残差块。初始化下采样层
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
#如果步长或通道数不匹配，则创建下采样层。
        layers = []#初始化一个空列表来存储残差块。
        layers.append(block(self.inplanes, planes, stride, downsample))#添加第一个残差块。
        self.inplanes = planes * block.expansion#更新输入通道数。
        for i in range(1, blocks):#添加剩余的残差块。
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)#返回一个序列化的残差块列表。

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x1 = self.layer3_1(x)
        x1 = self.layer4_1(x1)

        x2 = self.layer3_2(x)
        x2 = self.layer4_2(x2)

        return x1, x2#返回两个分支的输出。

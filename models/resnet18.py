import torch.nn as nn


def conv3x3(in_channels, out_channels, stride, dilation=1):
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=3,
        stride=stride,
        dilation=dilation,
        padding=dilation,
        bias=False,
    )


class block(nn.Module):
    """
    Basic Block: 3x3 Conv -> Batch Norm 1 -> ReLU -> 3x3 Conv -> Batch Norm 2 -> += initial -> ReLU
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super(block, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels, stride=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # add another layer if channel
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.ReLU(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # skip connection / identity matching
        out += self.shortcut(x)
        out = self.ReLU(out)

        return out


class resnet18(nn.Module):
    def __init__(self, num_classes=100, stride=1):
        super(resnet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.ReLU = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        self.layer4 = self.make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, out_channels, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.ReLU(x)
        x = self.maxpool(x)

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

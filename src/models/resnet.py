import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self, 
        c_in, 
        c_out, 
        stride=1,
        padding: int = 1,
        groups: int = 32,
    ):
        super(BasicBlock, self).__init__()
        self.exapnsion = BasicBlock.expansion

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        out = F.relu(x)
        return out


class BottleNeck(nn.Module):
    expansion = 4

    def __init__(
        self, 
        c_in, 
        c_out, 
        stride: int = 1,
        padding: int = 1,
        groups: int = 1,
    ):
        super(BottleNeck, self).__init__()
        self.exapnsion = BottleNeck.expansion

        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv3 = nn.Conv2d(c_out, c_out * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out * self.expansion)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out * self.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out * self.expansion),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x += identity 
        out = F.relu(x)
        return out


class ResNet(nn.Module):
    """
    Deep Residual Learning for Image Recognition (He et al., CVPR 2016)
    paper: https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf
    """
    def __init__(
        self,
        block, 
        num_blocks: list,
        num_strides: list, 
        c_init: int = 3,
        groups: int = 1,
        num_classes: int = 10,
    ):
        super(ResNet, self).__init__()
        
        self.c_in = 64

        self.stem_conv = nn.Conv2d(c_init, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(64, block, num_blocks[0], num_strides[0])
        self.layer2 = self._make_layer(128, block, num_blocks[1], num_strides[1])
        self.layer3 = self._make_layer(256, block, num_blocks[2], num_strides[2])
        self.layer4 = self._make_layer(512, block, num_blocks[3], num_strides[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
    def _make_layer(self, c_out, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.c_in, c_out, stride=stride))
            self.c_in = c_out * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)    
        out = self.fc(x)
       
        return out


class ResNet_S(nn.Module):                                  # s: small
    def __init__(
        self, 
        block,
        num_blocks: list,
        c_init: int = 3,
        groups:int = 1,
        num_classes: int = 1000,
    ):
        super(ResNet_S, self).__init__()

        self.c_in = 16

        self.stem_conv = nn.Conv2d(c_init, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self.__make__layer(16, block, num_blocks[0], stride=1)
        self.layer2 = self.__make__layer(32, block, num_blocks[1], stride=2)
        self.layer3 = self.__make__layer(64, block, num_blocks[2], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(64 * block.expansion, num_classes)

    def __make__layer(self, c_out, block, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.c_in, c_out, stride=stride))
            self.c_in = c_out * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], [1, 2, 2, 2], num_classes=num_classes)

def resnet20(num_classes):
    return ResNet_S(BasicBlock, [3,3,3], num_classes=num_classes)

def resnet20_greyscale(num_classes):
    return ResNet_S(BasicBlock, [3,3,3], c_init=1, num_classes=num_classes)

def resnet32(num_classes):
    return ResNet_S(BasicBlock, [5,5,5], num_classes=num_classes)

def resnet34(num_classes):
    return ResNet(BasicBlock, [3, 4, 6, 3], [1, 2, 2, 2], num_classes=num_classes)

def basic_resnet50(num_classes):
    return ResNet(BasicBlock, [3, 4, 14, 3], [1, 2, 2, 2], num_classes=num_classes)

def resnet50(num_classes):
    return ResNet(BottleNeck, [3, 4, 6, 3], [1, 2, 2, 2], num_classes=num_classes)

def resnet101(num_classes):
    return ResNet(BottleNeck, [3, 4, 23, 3], [1, 2, 2, 2], num_classes=num_classes)

def resnet152(num_classes):
    return ResNet(BottleNeck, [3, 8, 36, 3], [1, 2, 2, 2], num_classes=num_classes)


if __name__ =="__main__":
    num_classes = 10
    model = resnet34(num_classes)
    print(model(torch.randn((1, 3, 224, 224), dtype=torch.float32)))
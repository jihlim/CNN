import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleNetBlock(nn.Module):
    expansion = 4
    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        groups: int = 1,
        apply_gconv: bool = True
    ):
        super(ShuffleNetBlock, self).__init__()
        
        self.stride = stride
        self.groups = groups
        if stride != 1:
            c_out -= c_in// self.expansion
        c_mid = c_out
        
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1, groups=groups)
        if not apply_gconv:
            self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out * self.expansion, kernel_size=1, stride=1, groups=groups)
        self.bn3 = nn.BatchNorm2d(c_out * self.expansion)
        self.shortcut = nn.Identity()
        if stride != 1: 
            self.shortcut = nn.Sequential(
                nn.AvgPool2d((3,3), stride=stride, padding=1)
            )

    def _channel_shuffle(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.groups, -1, h, w)
        x = x.transpose(1,2)
        x = x.flatten(1,2)
        return x
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self._channel_shuffle(x)
        x = self.bn2(self.conv2(x))
        x = self.bn3(self.conv3(x))
        if self.stride == 1:
            x += identity
        else:
            x = torch.cat((x, identity), dim=1)
        out = F.relu(x)
        return out

            
class ShuffleNet(nn.Module):
    """
    ShuffleNet: An Extremely Efficient Convolutional Neural Networks for Mobile Devices (Zhang et al., CVPR 2018)
    paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_ShuffleNet_An_Extremely_CVPR_2018_paper.pdf
    """
    def __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        num_strides: list,
        c_init: int = 3,
        groups: int = 1,
        num_classes : int = 1000,
    ):
        super(ShuffleNet, self).__init__()
        assert len(channels) == 4, \
            "The number of channels element should be 4"
        assert len(num_blocks) == 3, \
            "The number of channels element should be 3"
        assert len(num_strides) == 3, \
            "The number of channels element should be 3"
        
        self.c_in = channels[0]
        self.groups = groups
        
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], num_strides[0], groups, apply_gconv=False)
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], num_strides[1], groups)
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], num_strides[2], groups)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        groups: int,
        apply_gconv: bool = True
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride, groups, apply_gconv))
            self.c_in = c_out * block.expansion
        return layer 
    
    def _channel_shuffle(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.groups, -1, h, w)
        x = x.transpose(1,2)
        x = x.flatten(1,2)
        return x
    
    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        # x = self.max_pool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def shufflenet_g1(num_classes):
    return ShuffleNet(ShuffleNetBlock, [24, 36, 72, 144], [4, 8, 4], [2, 2, 2], groups=1, num_classes=num_classes)

def shufflenet_g2(num_classes):
    return ShuffleNet(ShuffleNetBlock, [24, 50, 100, 200], [4, 8, 4], [2, 2, 2], groups=2, num_classes=num_classes)

def shufflenet_g3(num_classes):
    return ShuffleNet(ShuffleNetBlock, [24, 60, 120, 240], [4, 8, 4], [2, 2, 2], groups=3, num_classes=num_classes)

def shufflenet_g4(num_classes):
    return ShuffleNet(ShuffleNetBlock, [24, 68, 136, 272], [4, 8, 4], [2, 2, 2], groups=4, num_classes=num_classes)

def shufflenet_g8(num_classes):
    return ShuffleNet(ShuffleNetBlock, [24, 96, 192, 384], [4, 8, 4], [2, 2, 2], groups=8, num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ShuffleNet(ShuffleNetBlock, [24, 36, 72, 144], [4, 8, 4], [2, 2, 2], groups=1, num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
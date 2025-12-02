import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        c_out,
        ratio=16,
    ):
        super(SqueezeExcitation, self).__init__()
        
        c_over_r = c_out // ratio
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se_fc1 = nn.Conv2d(c_out, c_over_r, kernel_size=1)
        self.se_fc2 = nn.Conv2d(c_over_r, c_out, kernel_size=1)
        
    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = F.relu(self.se_fc1(scale))
        scale = torch.sigmoid(self.se_fc2(scale))
        x *= scale
        return x


class RegNetBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int = 32,
        expansion: int = 4,
        ratio: int = 4,
        add_se: bool = False,
    ):
        super(RegNetBlock, self).__init__()
        self.add_se = add_se
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.squeeze_exitation = SqueezeExcitation(c_out, ratio=ratio)
        self.conv3 = nn.Conv2d(c_out, c_out * expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out * expansion)
        self.activation = nn.ReLU()
        if add_se:
            self.activation = nn.SiLU()
        self.shortcut = nn.Identity()
        if stride !=1 or c_in != c_out * expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out * expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out * expansion),
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.add_se:
            x = self.squeeze_excitation(x)
        x = self.activation(x)
        x = self.bn3(self.conv3(x))
        x += identity
        out = self.activation(x)
        return out


class RegNet(nn.Module):
    """
    Designing Network Design Spaces (Radosavovic et al., CVPR 2020)
    paper: https://arxiv.org/pdf/2003.13678 
    
    """
    def  __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        c_init: int = 3,
        group_width: int = 8,
        expansion: int = 1,
        num_classes: int = 1000,
        add_se: bool = False,
    ):
        super(RegNet, self).__init__()
        
        assert len(channels) == 5, \
            "channels should have 5 elements"
        assert len(num_blocks) == 4, \
            "num_blocks should have 4 elements"
        
        cardinality = [c//group_width for c in channels[1:]]
        self.add_se = add_se
        self.c_in = channels[0]
        
        # Stem
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.activation = nn.ReLU()
        if add_se:
            self.activation = nn.SiLU()
        
        # Body
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], stride=1, groups=cardinality[0], expansion=1) 
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], stride=2, groups=cardinality[1], expansion=1)
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], stride=2, groups=cardinality[2], expansion=1) 
        self.layer4 = self._make_layer(block, channels[4], num_blocks[3], stride=2, groups=cardinality[3], expansion=1)
        
        # Head
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self, 
        block, 
        c_out, 
        num_blocks, 
        stride, 
        groups, 
        expansion
    ):
        strides = [stride] + [1] * (num_blocks -1)
        groups = int(max(1, groups))
        
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride=stride, groups=groups, expansion=expansion, add_se=self.add_se))
            self.c_in = c_out * expansion
        return layer

    def forward(self, x):
        # Stem
        x = self.activation(self.bn1(self.stem_conv(x)))
        
        # Body 
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Head
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out        

# RegNetX
def regnetx_200M(num_classes):
    # 41 Layers
    return RegNet(RegNetBlock, [24, 24, 56, 152, 368], [1, 1, 4, 7], group_width=8, num_classes=num_classes)

def regnetx_400M(num_classes):
    # 68 Layers
    return RegNet(RegNetBlock, [24, 32, 64, 160, 384], [1, 2, 7, 12], group_width=16, num_classes=num_classes)

def regnetx_600M(num_classes):
    # 50 Layers
    return RegNet(RegNetBlock, [48, 48, 96, 240, 528], [1, 3, 5, 7], group_width=24, num_classes=num_classes)

def regnetx_800M(num_classes):
    # 50 Layers
    return RegNet(RegNetBlock, [56, 64, 128, 288, 672], [1, 3, 7, 5], group_width=16, num_classes=num_classes)

def regnetx_1_6G(num_classes):
    # 56 Layers
    return RegNet(RegNetBlock, [80, 72, 168, 408, 912], [2, 4, 10, 2], group_width=24, num_classes=num_classes)

def regnetx_3_2G(num_classes):
    # 77 Layers
    return RegNet(RegNetBlock, [88, 96, 192, 432, 1008], [2, 6, 15, 2], group_width=48, num_classes=num_classes)

def regnetx_4G(num_classes):
    # 71 Layers
    return RegNet(RegNetBlock, [96, 80, 240, 560, 1360], [2, 5, 14, 2], group_width=40, num_classes=num_classes)

def regnetx_6_4G(num_classes):
    # 53 Layers
    return RegNet(RegNetBlock, [184, 168, 392, 784, 1624], [2, 4, 10, 1], group_width=56, num_classes=num_classes)

def regnetx_8G(num_classes):
    # 71 Layers
    return RegNet(RegNetBlock, [80, 80, 240, 720, 1920], [2, 5, 15, 1], group_width=120, num_classes=num_classes)

def regnetx_12G(num_classes):
    # 59 Layers
    return RegNet(RegNetBlock, [168, 224, 448, 896, 2240], [2, 5, 11, 1], group_width=112, num_classes=num_classes)

def regnetx_16G(num_classes):
    # 68 Layers
    return RegNet(RegNetBlock, [216, 256, 512, 896, 2048], [2, 6, 13, 1], group_width=128, num_classes=num_classes)

def regnetx_32G(num_classes):
    # 71 Layers
    return RegNet(RegNetBlock, [320, 336, 672, 1344, 2520], [2, 7, 13, 1], group_width=168, num_classes=num_classes)

# RegNetY
def regnety_200M(num_classes):
    # 41 Layers
    return RegNet(RegNetBlock, [24, 24, 56, 152, 368], [1, 1, 4, 7], group_width=8, num_classes=num_classes, add_se=True)

def regnety_400M(num_classes):
    # 50 Layers
    return RegNet(RegNetBlock, [48, 48, 104, 208, 440], [1, 3, 6, 6], group_width=8, num_classes=num_classes, add_se=True)

def regnety_600M(num_classes):
    # 47 Layers
    return RegNet(RegNetBlock, [48, 48, 112, 256, 608], [1, 3, 7, 4], group_width=16, num_classes=num_classes, add_se=True)

def regnety_800M(num_classes):
    # 44 Layers
    return RegNet(RegNetBlock, [56, 64, 128, 320, 768], [1, 3, 8, 2], group_width=16, num_classes=num_classes, add_se=True)

def regnety_1_6G(num_classes):
    # 83 Layers
    return RegNet(RegNetBlock, [48, 48, 120, 336, 888], [2, 6, 17, 2], group_width=24, num_classes=num_classes, add_se=True)

def regnety_3_2G(num_classes):
    # 62 Layers
    return RegNet(RegNetBlock, [80, 72, 216, 576, 1512], [2, 5, 13, 1], group_width=24, num_classes=num_classes, add_se=True)

def regnety_4G(num_classes):
    # 68 Layers
    return RegNet(RegNetBlock, [96, 128, 192, 512, 1088], [2, 6, 12, 2], group_width=64, num_classes=num_classes, add_se=True)

def regnety_6_4G(num_classes):
    # 77 Layers
    return RegNet(RegNetBlock, [112, 144, 288, 576, 1296], [2, 7, 14, 2], group_width=72, num_classes=num_classes, add_se=True)

def regnety_8G(num_classes):
    # 53 Layers
    return RegNet(RegNetBlock, [192, 168, 448, 896, 2016], [2, 4, 10, 1], group_width=56, num_classes=num_classes, add_se=True)

def regnety_12G(num_classes):
    # 59 Layers
    return RegNet(RegNetBlock, [168, 224, 448, 896, 2240], [2, 5, 11, 1], group_width=112, num_classes=num_classes, add_se=True)

def regnety_16G(num_classes):
    # 56 Layers
    return RegNet(RegNetBlock, [200, 224, 448, 1232, 3024], [2, 4, 11, 1], group_width=112, num_classes=num_classes, add_se=True)

def regnety_32G(num_classes):
    # 62 Layers
    return RegNet(RegNetBlock, [232, 232, 696, 1392, 3712], [2, 5, 12, 1], group_width=232, num_classes=num_classes, add_se=True)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = RegNet(RegNetBlock, [80, 240, 720, 1920], [2, 5, 15, 1], group_width=120, num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    output = model(x)
    print(x)
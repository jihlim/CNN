import torch
import torch.nn as nn
import torch.nn.functional as F


class SqueezeExcitation(nn.Module):
    def __init__(
        self,
        c_out,
        ratio: int = 16,
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
        x = x * scale
        return x


class EfficientNetBlock(nn.Module):
    def __init__(
        self,
        c_in, 
        c_out,
        kernel_size, 
        stride, 
        padding: int = 1,
        groups: int = 32,
        expansion: int = 6,
        ratio: int = 4, 
    ):
        super(EfficientNetBlock, self).__init__()
        
        c_mid = c_in * expansion
        self.stride = stride
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.squeeze_excitation = SqueezeExcitation(c_mid, ratio=4)
        self.conv3 = nn.Conv2d(c_mid, c_out, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.silu(self.bn1(self.conv1(x)))
        x = F.silu(self.bn2(self.conv2(x)))
        x = self.squeeze_excitation(x)
        out = self.bn3(self.conv3(x))
        if self.stride == 1:
            out += identity
        return out
    

class EfficientNet(nn.Module):
    """
    EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks (M. Tan, Q. V. Le, 2019)
    paper: https://arxiv.org/pdf/1905.11946
    """
    def __init__(
        self, 
        block, 
        channels, 
        num_blocks, 
        num_strides,
        c_init: int = 3,
        groups: int = 1, 
        num_classes: int = 1000,
        dropout_p: float = 0.2
    ):
        super(EfficientNet, self).__init__()
        
        assert len(channels) == 9, \
            "channels should have 9 elemenets"
        assert len(num_blocks) == 7, \
            "num_blocks should have 7 elements"
        assert len(num_strides) == 7, \
            "num_strides should have 7 elements"
        
        self.c_in = channels[0]
        self.stem_conv = nn.Conv2d(c_init, self.c_in, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(self.c_in)
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], num_strides[0], k=3, expansion=1)
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], num_strides[1], k=3, expansion=6)
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], num_strides[2], k=5, expansion=6)
        self.layer4 = self._make_layer(block, channels[4], num_blocks[3], num_strides[3], k=3, expansion=6)
        self.layer5 = self._make_layer(block, channels[5], num_blocks[4], num_strides[4], k=5, expansion=6)
        self.layer6 = self._make_layer(block, channels[6], num_blocks[5], num_strides[5], k=5, expansion=6)
        self.layer7 = self._make_layer(block, channels[7], num_blocks[6], num_strides[6], k=3, expansion=6)
        self.final_conv = nn.Conv2d(channels[7], channels[-1], kernel_size=1)
        self.final_bn = nn.BatchNorm2d(channels[-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(self, block, c_out, num_blocks, stride, k, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = nn.Sequential()
        for stride in strides:
            if k == 3:
                layer.append(block(self.c_in, c_out, k, stride, padding=1, expansion=expansion))
            elif k == 5:
                layer.append(block(self.c_in, c_out, k, stride, padding=2, expansion=expansion))
            self.c_in = c_out
        return layer
    
    def forward(self, x):
        x = F.silu(self.bn1(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.silu(self.final_bn(self.final_conv(x)))
        x = self.avg_pool(x)
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out
    
def efficientnet_b0(num_classes):
    # 50 Layers
    return EfficientNet(EfficientNetBlock, [32, 16, 24, 40, 80, 112, 192, 320, 1280], [1, 2, 2, 3, 3, 4, 1], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes)

def efficientnet_b1(num_classes):
    # 71 Layers
    return EfficientNet(EfficientNetBlock, [32, 16, 24, 40, 80, 112, 192, 320, 1280], [2, 3, 3, 4, 4, 5, 2], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes)

def efficientnet_b2(num_classes):
    # 71 Layers
    return EfficientNet(EfficientNetBlock, [32, 16, 24, 48, 88, 120, 208, 352, 1408], [2, 3, 3, 4, 4, 5, 2], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.3)

def efficientnet_b3(num_classes):
    # 80 Layers
    return EfficientNet(EfficientNetBlock, [40, 24, 32, 48, 96, 136, 232, 384, 1536], [2, 3, 3, 5, 5, 6, 2], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.3)

def efficientnet_b4(num_classes):
    # 98 Layers
    return EfficientNet(EfficientNetBlock, [48, 24, 32, 56, 112, 160, 272, 448, 1792], [2, 4, 4, 6, 6, 8, 2], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.4)

def efficientnet_b5(num_classes):
    # 119 Layers
    return EfficientNet(EfficientNetBlock, [48, 24, 40, 64, 128, 176, 304, 512, 2048], [3, 5, 5, 7, 7, 9, 3], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.4)

def efficientnet_b6(num_classes):
    # 137 Layer
    return EfficientNet(EfficientNetBlock, [56, 32, 40, 72, 144, 200, 344, 576, 2304], [3, 6, 6, 8, 8, 11, 3], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.5)

def efficientnet_b7(num_classes):
    # 167 Layers
    return EfficientNet(EfficientNetBlock, [64, 32, 48, 80, 160, 224, 384, 640, 2560], [4, 7, 7, 10, 10, 13, 4], [1, 2, 2, 2, 1, 2, 1], num_classes=num_classes, dropout_p=0.5)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = EfficientNet(EfficientNetBlock, [32, 16, 24, 40, 80, 112, 192, 320, 1280], [1, 2, 2, 3, 3, 4, 1], [1, 2, 2, 2, 1, 2, 1], num_classes=10).to(device)
    x = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
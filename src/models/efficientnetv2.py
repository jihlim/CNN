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
        
        c_over_r = int(c_out // ratio)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se_fc1 = nn.Conv2d(c_out, c_over_r, kernel_size=1)
        self.se_fc2 = nn.Conv2d(c_over_r, c_out, kernel_size=1)
        
    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = F.relu(self.se_fc1(scale))
        scale = torch.sigmoid(self.se_fc2(scale))
        x = x * scale
        return x


class EfficientNetV2Block(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int,
        groups: int = 3,
        expansion: int = 4,
        ratio: int = 4,
        is_fused: bool = True,
        add_se: bool = False,
    ):
        super(EfficientNetV2Block, self).__init__()
        
        c_mid = c_in * expansion
        self.stride = stride
        self.add_se = not is_fused
        self.conv1 = nn.Sequential(
            nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1),
            nn.BatchNorm2d(c_mid),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        )
        if is_fused:
            self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.squeeze_excitation = SqueezeExcitation(c_mid, ratio=ratio)
        self.conv2 = nn.Conv2d(c_mid, c_out, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.silu(self.bn1(self.conv1(x)))
        if self.add_se:
            x = self.squeeze_excitation(x)
        out = self.bn2(self.conv2(x))
        if self.stride == 1:
            out += identity
        return out
        

class EfficientNetV2(nn.Module):
    """
    EfficientNetV2:  Smaller Models and Faster Training (M. Tan and Q. V. Le, arXiv 2021)
    paper: https://arxiv.org/pdf/2104.00298
    """
    def __init__(
        self,
        block, 
        channels: list,
        num_blocks: list,
        num_strides: list,
        k_sizes: list,
        expansion: list,
        is_fused: list,
        c_init: int = 3,
        groups: int = 1,
        ratio: int = 4,
        num_classes: int = 1000,
        dropout_p: float = 0.2
    ):
        super(EfficientNetV2, self).__init__()
        assert len(channels) == 8 or len(channels) == 9, \
            "channels should have 8 elemenets for base, S models, 9 for M, L, XL models"
        assert len(num_blocks) == 6 or len(num_blocks) == 7, \
            "num_blocks should have 6 elements for base, S models, 7 for M, L, XL models"
        assert len(num_strides) == 6 or len(num_strides) == 7, \
            "num_strides should have 6 elements for base, S models, 7 for M, L, XL models"
        assert len(k_sizes) == 6 or len(k_sizes) == 7, \
            "k_sizes should have 6 elements for base, S models, 7 for M, L, XL models"
        assert len(expansion) == 6 or len(expansion) == 7, \
            "expansion should have 6 elements for base, S models, 7 for M, L, XL models"
        assert len(is_fused) == 6 or len(is_fused) == 7, \
            "num_strides should have 6 elements for base, S models, 7 for M, L, XL models"
        
        self.c_in = channels[0]
        self.layer7_required = len(num_blocks) == 7
        
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], num_strides[0], k_sizes[0], expansion[0], ratio, is_fused[0])
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], num_strides[1], k_sizes[1], expansion[1], ratio, is_fused[1])
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], num_strides[2], k_sizes[2], expansion[2], ratio, is_fused[2])
        self.layer4 = self._make_layer(block, channels[4], num_blocks[3], num_strides[3], k_sizes[3], expansion[3], ratio, is_fused[3])
        self.layer5 = self._make_layer(block, channels[5], num_blocks[4], num_strides[4], k_sizes[4], expansion[4], ratio, is_fused[4])
        self.layer6 = self._make_layer(block, channels[6], num_blocks[5], num_strides[5], k_sizes[5], expansion[5], ratio, is_fused[5])
        if self.layer7_required:
            self.layer7 = self._make_layer(block, channels[7], num_blocks[6], num_strides[6], k_sizes[6], expansion[6], ratio, is_fused[6])
        self.final_conv = nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1)
        self.final_bn = nn.BatchNorm2d(channels[-1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout_p)
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        k: int,
        expansion: int, 
        ratio: int,
        is_fused: bool,
    ):
        strides = [stride] + [1] * (num_blocks -1)
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, kernel_size=k, stride=stride, expansion=expansion, ratio=ratio, is_fused=is_fused))
            self.c_in = c_out
        return layer
    
    def forward(self, x):
        x = F.silu(self.stem_bn(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        if self.layer7_required:
            x = self.layer7(x)
        x = F.silu(self.final_bn(self.final_conv(x)))
        x = self.avg_pool(x)
        # x = self.dropout(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def efficientnetv2_base(num_classes):
    # 61 Layers
    return EfficientNetV2(
        EfficientNetV2Block, 
        [24, 16, 32, 48, 96, 112, 192, 1280], 
        [1, 2, 2, 3, 5, 8], 
        [1, 2, 2, 2, 1, 2], 
        [3, 3, 3, 3, 3, 3], 
        [1, 4, 4, 4, 6, 6], 
        [True, True, True, False, False, False], 
        num_classes=num_classes
    )
  
def efficientnetv2_S(num_classes):
    # 113 Layers
    return EfficientNetV2(
        EfficientNetV2Block, 
        [24, 24, 48, 64, 128, 160, 256, 1280], 
        [2, 4, 4, 6, 9, 15], 
        [1, 2, 2, 2, 1, 2], 
        [3, 3, 3, 3, 3, 3], 
        [1, 4, 4, 4, 6, 6], 
        [True, True, True, False, False, False], 
        num_classes=num_classes
    )

def efficientnetv2_M(num_classes):
    # 155 Layers
    return EfficientNetV2(
        EfficientNetV2Block, 
        [24, 24, 48, 80, 160, 176, 304, 512, 1280], 
        [3, 5, 5, 7, 14, 18, 5], 
        [1, 2, 2, 2, 1, 2, 1], 
        [3, 3, 3, 3, 3, 3, 3], 
        [1, 4, 4, 4, 6, 6, 6], 
        [True, True, True, False, False, False, False], 
        num_classes=num_classes
    )

def efficientnetv2_L(num_classes):
    # 220 Layers
    return EfficientNetV2(
        EfficientNetV2Block, 
        [24, 32, 64, 96, 192, 224, 384, 640, 1280], 
        [4, 7, 7, 10, 19, 25, 7], 
        [1, 2, 2, 2, 1, 2, 1], 
        [3, 3, 3, 3, 3, 3, 3], 
        [1, 4, 4, 4, 6, 6, 6], 
        [True, True, True, False, False, False, False], 
        num_classes=num_classes
    )

def efficientnetv2_XL(num_classes):
    # 283 Layers
    return EfficientNetV2(
        EfficientNetV2Block, 
        [24, 32, 64, 96, 192, 256, 512, 640, 1280], 
        [4, 8, 8, 16, 24, 32, 8], 
        [1, 2, 2, 2, 1, 2, 1], 
        [3, 3, 3, 3, 3, 3, 3], 
        [1, 4, 4, 4, 6, 6, 6], 
        [True, True, True, False, False, False, False], 
        num_classes=num_classes
    )
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model =  EfficientNetV2(EfficientNetV2Block, [24, 24, 48, 64, 128, 160, 256, 1280], [2, 4, 4, 6, 9, 15], [1, 2, 2, 2, 1, 2], [3, 3, 3, 3, 3, 3], [1, 4, 4, 4, 6, 6], [True, True, True, False, False, False], num_classes=10).to(device)
    x = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape) 
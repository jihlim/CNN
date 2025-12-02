from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LargeSqueezeExcitation(nn.Module):
    def __init__(
        self,
        c_out: int,
        ratio: int = 4,
    ):
        super(LargeSqueezeExcitation, self).__init__()
        
        c_over_r = c_out // ratio
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se_fc1 = nn.Conv2d(c_out, c_over_r, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.se_fc2 = nn.Conv2d(c_over_r, c_out, kernel_size=1)
        self.hardsigmoid = nn.Hardsigmoid(inplace=True)
        
    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = self.fc1(scale)
        scale = self.relu(scale)
        scale = self.fc2(scale)
        scale = self.hardsigmoid(scale)
        out = x * scale
        return out


class MobileNetV3Block(nn.Module):
    def __init__(
        self, 
        c_in: int, 
        c_out: int, 
        kernel_size: int,
        stride: int,
        padding: int = 1,
        groups: int = 16,
        expansion: float = 6.0,
        add_se: bool = False,
        nonlinearity: str = "relu"
    ):
        super(MobileNetV3Block, self).__init__()
        
        c_mid = int(c_out * expansion)
        self.add_se = add_se
        self.stride = stride
        self.squeeze_excitation = LargeSqueezeExcitation()
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.activation = nn.ReLU(inplace=True)
        if nonlinearity == "hardswish":
            self.activation = nn.Hardswish(inplace=True)
        self.shortcut = nn.Identity()
        if c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=1),
                nn.BatchNorm2d(c_out),
            )
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = self.activation(self.bn1(self.conv1))
        x = self.activation(self.bn2(self.conv2))
        if self.add_se:
            x = self.squeeze_excitation(x)
        x = self.bn3(self.conv3(x))
        if self.stride == 1:
            x += identity
        out = self.activation(x)
        return out
             

# class MobielNetV3(nn.Module):
#     r"""
#     Searching for MobileNetV3 (Howard et al., ICCV 2019)
#     paper: https://openaccess.thecvf.com/content_ICCV_2019/papers/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.pdf
#     """
#     def __init__(
#         self,
#         block,
#         channels: list,
#         num_blocks: list,
#         num_strides: list,
#         c_init: int = 3,
#         groups: int = 1,
#         nonlinearity: str = "relu",
#         num_classes: int = 1000
#     ):
#         super(MobielNetV3, self).__init__()
        
#         self.c_in = channels[0]
#         self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels[0])
#         self.activation = nn.Hardswish(inplace=True)
#         self.layer1 = _make_layer(block, channels[0], num_blocks[], strides=2, k=, expansion=, nonlinearity="relu")
#         self.layer2 = _make_layer(block, channels[1], num_blocks[], strides=2, k=, expansion=, nonlinearity="relu")
#         self.layer3 = _make_layer(block, channels[2], num_blocks[], strides=2, k=, expansion=, nonlinearity=)
#         self.layer4 = _make_layer(block, channels[3], num_blocks[], strides=1, k=, expansion=, nonlinearity="hardswish")
#         self.layer5 = _make_layer(block, channels[4], num_blocks[], strides=2, k=, expansion=, nonlinearity="hardswish")
#         self.layer6 = _make_layer(block, channels[], num_blocks[], strides=1, k=, expansion=, nonlinearity="hardswish")
#         self.conv1 = nn.Conv2d(channels[], channels[], kernel_size=1, stirde=1)
#         self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
#         self.conv2 = nn.Conv2d(channels[], channels[], kernel_size=1, stride=1)
#         self.final_conv = nn.Conv2d(channels[-1], num_classes, kernel_size=1, stride=1)
        
#     def _make_layer(
#         self,
#         block,
#         c_out: int,
#         num_blocks: int,
#         stride: int,
#         k: int,
#         expansion: int,
#         nonlinearity: str,
#     ):
#         strides = [1] * (num_blocks -1 ) + [stride]
#         layer = nn.Sequential()
        
#         for stride in strides:
#             if k == 3:
#                 layer.append(block(self.c_in, c_out, kernel_size=3, stride=stride, padding=1, add_se=, expnasion=expansion, nonlinearity=nonlinearity))
#             if k == 5:
#                 layer.append(block(self.c_in, c_out, kernel_size=5, stride=stride, padding=2, add_se=, expnasion=expansion, nonlinearity=nonlinearity))
#             self.c_in = c_out
#         return layer
    
#     def forward(self, x):
#         x = self.activation(self.bn1(self.stem_conv(x)))
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.layer5(x)
#         x = self.layer6(x)
#         x = self.activation((self.conv1(x)))
#         x = self.avg_pool(x)
#         x = self.activation(self.conv2(x))
#         x = self.final_conv(x)
#         out = x.view(x.size(0), -1)
#         return out


class MobileNetV3Large(nn.Module):
    """
    MobileNetV3-Large
    49 Layers (45 + 4)
    """
    def __init__(
        self,
        block,
        c_init: int = 3,
        num_classes: int = 1000,
    ):
        super(MobileNetV3Large, self).__init__()
    
        self.stem_conv = nn.Conv2d(c_init, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.Hardswish(inplace=True)
        
        self.layer = nn.Sequential( 
            # Layer1: -> 6 
            block(16,   16,     kernel_size=3,  stride=1, padding=1,    expansion=1.0,  add_se=False, nonlinearity="relu"),
            block(16,   24,     kernel_size=3,  stride=2, padding=1,    expansion=4.0,  add_se=False, nonlinearity="relu"),
            
            # Layer2 -> 6
            block(24,   24,     kernel_size=3,  stride=1, padding=1,    expansion=3.0,  add_se=False, nonlinearity="relu"),
            block(24,   40,     kernel_size=5,  stride=2, padding=2,    expansion=3.0,  add_se=True,  nonlinearity="relu"),
            
            # Layer3 -> 9
            block(40,   24,     kernel_size=5,  stride=1, padding=2,    expansion=3.0,  add_se=True,  nonlinearity="relu"),
            block(40,   16,     kernel_size=5,  stride=1, padding=2,    expansion=3.0,  add_se=True,  nonlinearity="relu"),
            block(40,   24,     kernel_size=3,  stride=2, padding=1,    expansion=6.0,  add_se=False, nonlinearity="hardswish"),
            
            # Layer4 -> 12
            block(80,   80,     kernel_size=3,  stride=1, padding=1,    expansion=2.5,  add_se=False, nonlinearity="hardswish"),
            block(80,   80,     kernel_size=3,  stride=1, padding=1,    expansion=2.3,  add_se=False, nonlinearity="hardswish"),
            block(80,   80,     kernel_size=3,  stride=1, padding=1,    expansion=2.3,  add_se=False, nonlinearity="hardswish"),
            block(80,   112,    kernel_size=3,  stride=1, padding=1,    expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            
            # Layer5 -> 6
            block(112,  112,    kernel_size=3,  stride=1, padding=1,    expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            block(112,  24,     kernel_size=5,  stride=2, padding=2,    expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            
            # Layer6 -> 6
            block(160,  160,    kernel_size=5,  stride=1, padding=2,    expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            block(160,  160,    kernel_size=5,  stride=1, padding=2,    expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
        )
        
        self.conv1 = nn.Conv2d(160, 960, kernel_size=1, stirde=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv2 = nn.Conv2d(960, 1280, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(1280, num_classes, kernel_size=1, stride=1) 
    
    def forward(self, x):
        x = self.activation(self.bn1(self.stem_conv(x)))
        x = self.layer(x)
        x = self.activation(self.conv1(x))
        x = self.avg_pool(x)
        x = self.activation(self.conv2(x))
        x = self.final_conv(x)
        out = x.view(x.size(0), -1)
        return out


class MobileNetV3Small(nn.Module):
    """
    MobileNetV3-Small
    37 Layers (33 + 4)
    """
    def __init__(
        self,
        block,
        c_init: int = 3,
        num_classes: int = 1000,
    ):
        super(MobileNetV3Small, self).__init__()
    
        self.stem_conv = nn.Conv2d(c_init, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.activation = nn.Hardswish(inplace=True)
        
        self.layer = nn.Sequential( 
            # Layer1 -> 3
            block(16, 16, kernel_size=3, stride=2, padding=1,   expansion=1.0,  add_se=True,  nonlinearity="relu"),
            
            # Layer2 -> 3
            block(16, 24, kernel_size=3, stride=2, padding=1,   expansion=4.5,  add_se=False, nonlinearity="relu"),
            
            # Layer3 -> 6
            block(24, 24, kernel_size=3, stride=1, padding=1,   expansion=3.67, add_se=False, nonlinearity="relu"),
            block(24, 40, kernel_size=5, stride=2, padding=2,   expansion=4.0,  add_se=True,  nonlinearity="hardswish"),
            
            # Layer4 -> 9
            block(40, 40, kernel_size=5, stride=1, padding=2,   expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            block(40, 40, kernel_size=5, stride=1, padding=2,   expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            block(40, 48, kernel_size=5, stride=1, padding=2,   expansion=3.0,  add_se=True,  nonlinearity="hardswish"),
            
            # Layer5 -> 6
            block(48, 48, kernel_size=5, stride=1, padding=2,   expansion=3.0,  add_se=True,  nonlinearity="hardswish"),
            block(48, 96, kernel_size=5, stride=2, padding=2,   expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            
            # Layer6 -> 6
            block(96, 96, kernel_size=5, stride=1, padding=2,   expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
            block(96, 96, kernel_size=5, stride=1, padding=2,   expansion=6.0,  add_se=True,  nonlinearity="hardswish"),
        )
        
        self.conv1 = nn.Conv2d(96, 576, kernel_size=1, stirde=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.conv2 = nn.Conv2d(576, 1024, kernel_size=1, stride=1)
        self.final_conv = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1) 
    
    def forward(self, x):
        x = self.activation(self.bn1(self.stem_conv(x)))
        x = self.layer(x)
        x = self.activation(self.conv1(x))
        x = self.avg_pool(x)
        x = self.activation(self.conv2(x))
        x = self.final_conv(x)
        out = x.view(x.size(0), -1)
        return out       

def mobilenetv3_small(num_classes):
    return MobileNetV3Small(MobileNetV3Block, [16, 16, 24, 40, 48, 96, 576, 1024], [1, 1, 2, 3, 2, 2], num_classes=num_classes)

def mobilenetv3_large(num_classes):
    return MobileNetV3Large(MobileNetV3Block, [16, 24, 40, 80, 112, 160, 960, 1280], [2, 2, 3, 4, 2, 2], num_classes=num_classes)

if __name__  == "__main__":
    device = torch.device("cuda" if torch.cuda.is_availabel() else "cpu")
    model = MobileNetV3Small(MobileNetV3Block, [16, 16, 24, 40, 48, 96, 576, 1024], [1, 1, 2, 3, 2, 2], num_classes=10).to(device)
    x = torch.randn((3, 224, 224), dtype=torch.float32, device=device)
    out = model(x)
    print(out.shape)
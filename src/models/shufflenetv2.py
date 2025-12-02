import torch
import torch.nn as nn
import torch.nn.functional as F


class ShuffleNetV2Block(nn.Module):
    expansion = 1
    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int = 1,
        groups: int = 1,
    ):
        super(ShuffleNetV2Block, self).__init__()
        
        self.stride = stride
        self.groups = groups
        
        c_mid = c_out - c_in
        if stride == 1:
            c_in //= 2
            c_mid = c_in
        
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_mid, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_mid)
        self.shortcut = nn.Identity()
        if stride != 1: 
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in),
                nn.BatchNorm2d(c_in),
                nn.Conv2d(c_in, c_in, kernel_size=1, stride=1),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True)
            )
    
    def _channel_split(self, x):
        x_0, x_1 = x.chunk(2, dim=1)
        return x_0, x_1
    
    def _channel_shuffle(self, x):
        b, c, h, w = x.shape
        x = x.view(b, self.groups, -1, h, w)
        x = x.transpose(1,2)
        x = x.flatten(1,2)
        return x
    
    def forward(self, x):
        x_0, x_1 = x, x
        if self.stride == 1:
            x_0, x_1 = self._channel_split(x)
        
        identity = self.shortcut(x_0)   
            
        # x = self.residual(x_1)
        x = F.relu(self.bn1(self.conv1(x_1)))
        x = self.bn2(self.conv2(x))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat((x, identity), dim=1)
        out = self._channel_shuffle(x)
        return out

            
class ShuffleNetV2(nn.Module):
    r"""
    ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design (Ma et al., ECCV 2018)
    paper: https://openaccess.thecvf.com/content_ECCV_2018/papers/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.pdf
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
        super(ShuffleNetV2, self).__init__()
        assert len(channels) == 5, \
            "The number of channels element should be 5"
        assert len(num_blocks) == 3, \
            "The number of channels element should be 3"
        assert len(num_strides) == 3, \
            "The number of channels element should be 3"
        
        self.c_in = channels[0]
        self.groups = groups
        
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], num_strides[0], groups)
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], num_strides[1], groups)
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], num_strides[2], groups)
        self.conv1 = nn.Conv2d(channels[3], channels[4], kernel_size=1, stride=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        groups: int,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride, groups))
            self.c_in = c_out
        return layer 
    
    def forward(self, x):
        x = self.activation(self.stem_bn(self.stem_conv(x)))
        # x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv1(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


class ShuffleNetV2Large(nn.Module):
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
        super(ShuffleNetV2Large, self).__init__()
        assert len(channels) == 6 or len(channels) == 7, \
            "The number of channels element should be 6 for 0.5x, 1x, 1.5x, 2x models, 7 for ShufflenNet v2-50 model"
        assert len(num_blocks) == 4 or len(num_blocks) == 5, \
            "The number of channels element should be 4 for 0.5x, 1x, 1.5x, 2x models, 5 for ShufflenNet v2-50 model"
        assert len(num_strides) == 4 or len(num_strides) == 5, \
            "The number of channels element should be 4 for 0.5x, 1x, 1.5x, 2x models, 5 for ShufflenNet v2-50 model"
        
        self.c_in = channels[0]
        self.groups = groups
        self.layer4_required = len(num_blocks) == 5
        
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[1], num_blocks[0], num_strides[0], groups)
        self.layer2 = self._make_layer(block, channels[2], num_blocks[1], num_strides[1], groups)
        self.layer3 = self._make_layer(block, channels[3], num_blocks[2], num_strides[2], groups)
        if self.layer4_required:
            self.layer4 = self._make_layer(block, channels[4], num_blocks[3], num_strides[3], groups)
        self.conv1 = nn.Conv2d(channels[-2], channels[-1], kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(channels[-1])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        groups: int,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride, groups))
            self.c_in = c_out
        return layer 
    
    def forward(self, x):
        x = self.activation(self.stem_bn(self.stem_conv(x)))
        # x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.layer4_required:
            x = self.layer4(x)
        x = self.activation(self.bn1(self.conv1(x)))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out


def shufflenetv2_0_5x(num_classes):
    # 51 Layers
    return ShuffleNetV2(ShuffleNetV2Block, [24, 48, 96, 192, 1024], [4, 8, 4], [2, 2, 2], groups=2, num_classes=num_classes)

def shufflenetv2_1x(num_classes):
    # 51 Layers
    return ShuffleNetV2(ShuffleNetV2Block, [24, 116, 232, 464, 1024], [4, 8, 4], [2, 2, 2], groups=2, num_classes=num_classes)

def shufflenetv2_1_5x(num_classes):
    # 51 Layers
    return ShuffleNetV2(ShuffleNetV2Block, [24, 176, 352, 704, 1024], [4, 8, 4], [2, 2, 2], groups=2, num_classes=num_classes)

def shufflenetv2_2x(num_classes):
    # 51 Layers
    return ShuffleNetV2(ShuffleNetV2Block, [24, 244, 488, 976, 2048], [4, 8, 4], [2, 2, 2], groups=2, num_classes=num_classes)

def shufflenetv2_50(num_classes):
    # 51 Layers
    return ShuffleNetV2Large(ShuffleNetV2Block, [64, 244, 488, 976, 1952, 2048], [3, 4, 6, 3], [1, 2, 2, 2], groups=2, num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ShuffleNetV2(ShuffleNetV2Block, [24, 48, 96, 192, 1024], [4, 8, 4], [2, 2, 2], groups=3, num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F 


class SeparableConv(nn.Module):
    def __init__(
        self,
        c_in,
        c_out, 
        stride: int = 1,
        padding:int = 1,
        groups: int = 1,
        init_activation: bool = True
    ):
        super(SeparableConv, self).__init__()
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, groups=c_out)
        self.bn2 = nn.BatchNorm2d(c_out)
        
    def forward(self, x):
        x = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(x))
        return out


class XceptionBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_mid,
        c_out, 
        stride: int = 1,
        padding: int = 1,
        groups: int = 32,
        init_activation: bool = True,
    ):
        super(XceptionBlock, self).__init__()
        
        self.stride = stride
        self.init_activation = init_activation
        self.separable_conv1 = SeparableConv(c_in, c_mid)
        self.separable_conv2 = SeparableConv(c_mid, c_out)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        # self.shortcut = nn.Identity()
        # if stride != 1: 
        self.shortcut = nn.Sequential(
            nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
            nn.BatchNorm2d(c_out)
        )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        if self.init_activation:  
            x = F.relu(x)     
        x = self.separable_conv1(x)
        x = F.relu(x)
        x = self.separable_conv2(x)
        if self.stride != 1:
            x = self.max_pool(x)
        out = x + identity
        return out


class _XceptionMiddleBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_mid,
        c_out,
        stride: int = 1,
        groups: int = 1,
        init_activation: bool = True
    ):
        super(_XceptionMiddleBlock, self).__init__()
        
        self.separable_conv1 = SeparableConv(c_in, c_mid)
        self.separable_conv2 = SeparableConv(c_mid, c_mid)
        self.separable_conv3 = SeparableConv(c_mid, c_out)
        self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(x)
        x = self.separable_conv1(x)
        x = F.relu(x)
        x = self.separable_conv2(x)
        x = F.relu(x)
        x = self.separable_conv3(x)
        out = x + identity
        return out

      
class Xception(nn.Module):
    r"""
    Xception: Deep Learning with Depthwise Separable Convolutions (F. Chollet, CVPR 2017)
    paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Chollet_Xception_Deep_Learning_CVPR_2017_paper.pdf
    """
    def __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        num_strides: list,
        c_init: int = 3, 
        num_classes: int = 1000,
    ):
        super(Xception, self).__init__()
        
        self.c_in = channels[1]
        
        # Entry flow
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_bn = nn.BatchNorm2d(channels[0])
        self.conv1 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[1])
        self.layer1 = self._make_layer(block, channels[2], channels[2], num_blocks[0], num_strides[0], init_activation=False) 
        self.layer2 = self._make_layer(block, channels[3], channels[3], num_blocks[1], num_strides[1])
        self.layer3 = self._make_layer(block, channels[4], channels[4], num_blocks[2], num_strides[2])
        
        # Middle flow
        self.layer4 = self._make_layer(_XceptionMiddleBlock, channels[4], channels[4], num_blocks[3], num_strides[3]) 
        
        # Exit flow
        self.layer5 = self._make_layer(block, channels[4], channels[5], num_blocks[4], num_strides[4])
        self.separable_conv1 = SeparableConv(channels[5], channels[6])
        self.separable_conv2 = SeparableConv(channels[6], channels[7])
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self,
        block,
        c_mid: int,
        c_out: int,
        num_blocks: int,
        stride:int,
        init_activation:bool = True,
    ):  
        layer = nn.Sequential()
        for _ in range(num_blocks):
            layer.append(block(self.c_in, c_mid, c_out, stride=stride, init_activation=init_activation))
            self.c_in = c_out
        return layer
            
    def forward(self, x):
        x = F.relu(self.stem_bn(self.stem_conv(x)))
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = F.relu(self.separable_conv1(x))
        x = F.relu(self.separable_conv2(x))
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out 


def xception71(num_classes):
    return Xception(XceptionBlock, [32, 64, 128, 256, 728, 1024, 1536, 2048], [1, 1, 1, 8, 1], [2, 2, 2, 1, 2], num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Xception(XceptionBlock, [32, 64, 128, 256, 728, 1024, 1536, 2048], [1, 1, 1, 8, 1], [2, 2, 2, 1, 2], num_classes=10).to(device)
    x = torch.randn((1, 3, 128, 128), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
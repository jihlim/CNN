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


class SepConv(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int, 
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        groups: int =1
    ):
        super(SepConv, self).__init__()
        
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=1, padding=1, groups=c_in)
        self.bn1 = nn.BatchNorm2d(c_in)
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride)
        self.bn2 = nn.BatchNorm2d(c_out)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(x))
        return out


class MnasNetBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size: int,
        stride: int = 1,
        padding: int =1,
        groups: int = 1,
        expansion: int = 6,
        add_se: bool = False
    ):
        super(MnasNetBlock, self).__init__()
        
        self.add_se = add_se
        
        self.conv1 = nn.Conv2d(c_in, c_out * expansion, kernel_size= 1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_out * expansion)
        self.conv2 = nn.Conv2d(c_out * expansion, c_out * expansion, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_out * expansion)
        self.bn2 = nn.BatchNorm2d(c_out * expansion)
        self.squeeze_excitation = SqueezeExcitation(c_out * expansion, ratio=4)
        self.conv3 = nn.Conv2d(c_out * expansion, c_out, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out)
        self.shortcut = nn.Identity()
        if stride != 1 or c_in != c_out:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out)
            )
        
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        if self.add_se:
            x = self.squeeze_excitation(x)
        x = self.bn3(self.conv3(x))
        out = x + identity
        return out
    

class MnasNet(nn.Module):
    """
    MnasNet: Platform-Aware Neural Architecture Search for Mobile (Tan et al., CVPR 2019)
    paper: https://openaccess.thecvf.com/content_CVPR_2019/papers/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.pdf
    """
    def __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        num_strides: list,
        k_sizes: list,
        c_init: int = 3,
        num_classes = 1000,
    ):
        super(MnasNet, self).__init__()
        
        assert len(channels) == 8, \
            "Channels should have 8 elements"
        assert len(num_blocks) == 6, \
            "num_blocks should have 6 elements"
        assert len(num_strides) == 6, \
            "num_strides should have 6 elements"
        assert len(k_sizes) == 6, \
            "k_sizes should have 6 elements"
            
        self.c_in = channels[1]
        
        self.stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.sep_conv = SepConv(channels[0], channels[1])
        self.layer1 = self._make_layer(block, channels[2], num_blocks[0], num_strides[0], k_sizes[0], expansion=6, add_se=False)
        self.layer2 = self._make_layer(block, channels[3], num_blocks[1], num_strides[1], k_sizes[1], expansion=3, add_se=True)
        self.layer3 = self._make_layer(block, channels[4], num_blocks[2], num_strides[2], k_sizes[2], expansion=6, add_se=False)
        self.layer4 = self._make_layer(block, channels[5], num_blocks[3], num_strides[3], k_sizes[3], expansion=6, add_se=True)
        self.layer5 = self._make_layer(block, channels[6], num_blocks[4], num_strides[4], k_sizes[4], expansion=6, add_se=True)
        self.layer6 = self._make_layer(block, channels[7], num_blocks[5], num_strides[5], k_sizes[5], expansion=6, add_se=False)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        k: int,
        expansion: int = 6,
        add_se: bool = False,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        
        layer = nn.Sequential()
        for stride in strides:
            if k == 3:
                layer.append(block(self.c_in, c_out, kernel_size=k, stride=stride, padding=1, expansion=expansion, add_se=add_se))
            elif k == 5:
                layer.append(block(self.c_in, c_out, kernel_size=k, stride=stride, padding=2, expansion=expansion, add_se=add_se))
            self.c_in = c_out
        return layer
    
    def forward(self, x):
        x = self.stem_conv(x)
        x = self.sep_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def mnasnet_a1(num_classes):
    return MnasNet(MnasNetBlock, [32, 16, 24, 40, 80, 112, 160, 320], [2, 3, 4, 2, 3, 1], [2, 2, 2, 1, 2, 1], [3, 5, 3, 3, 5, 3], num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model =  MnasNet(MnasNetBlock, [32, 16, 24, 40, 80, 112, 160, 320], [2, 3, 4, 2, 3, 1], [2, 2, 2, 1, 2, 1], [3, 5, 3, 3, 5, 3], num_classes=10).to(device)
    x = torch.randn((1, 3, 229, 229), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
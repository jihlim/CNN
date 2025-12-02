import torch
import torch.nn as nn
import torch.nn.functional as F


class ResNeXtBlock(nn.Module):
    expansion = 2
    def __init__(
        self, 
        c_in, 
        c_out, 
        stride=1, 
        groups=32,
    ):
        super(ResNeXtBlock, self).__init__()
        
        c_mid = c_out
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.bn2 = nn.BatchNorm2d(c_mid)
        self.conv3 = nn.Conv2d(c_mid, c_out * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out * self.expansion)
        self.shortcut = nn.Identity()
        if c_in != c_out * self.expansion or stride != 1:
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
    

class ResNeXt(nn.Module):
    r"""
    Aggregated Residual Transformations for Deep Neural Networks (Xie et al., CVPR 2017)
    paper: https://openaccess.thecvf.com/content_cvpr_2017/papers/Xie_Aggregated_Residual_Transformations_CVPR_2017_paper.pdf
    """
    def __init__(
        self,
        block, 
        channels, 
        num_blocks, 
        c_init=3,
        groups = 32,   
        num_classes=1000,
    ):
        super(ResNeXt, self).__init__()
        
        self.c_in = 64
        self.groups = groups
        self.stem_conv = nn.Conv2d(c_init, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[3] * block.expansion, num_classes)
        
    def _make_layer(self, block, c_out, num_block, stride):
        layers = nn.Sequential()
        strides = [stride] + [1] * (num_block -  1)
        for stride in strides:
            layers.append(block(self.c_in, c_out, stride, groups=self.groups))
            self.c_in = c_out * block.expansion
        return layers
    
    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        # x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x) 
        x = self.layer3(x) 
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out
    
def resnext26(num_classes):
    return ResNeXt(ResNeXtBlock, [128, 256, 512, 1024], [2, 2, 2, 2], num_classes=num_classes)

def resnext50(num_classes):
    return ResNeXt(ResNeXtBlock, [128, 256, 512, 1024], [3, 4, 6, 3], num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = ResNeXt(ResNeXtBlock,[128, 256, 512, 1024], [3, 4, 6, 3], num_classes=10).to(device)
    input = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(input)
    print(output)
    
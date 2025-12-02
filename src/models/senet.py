import torch
import torch.nn as nn
import torch.nn.functional as F 


class GlobalAvgPooling(nn.Module):
    def __init__(self,):
        super(GlobalAvgPooling, self).__init__()
    
    def forward(self,x):
        scale = torch.mean(x, dim=(2,3))
        return scale


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
    

class SENetBlock(nn.Module):
    expansion = 1
    def __init__(
        self,
        c_in,
        c_out,
        stride, 
        groups: int = 32,
        ratio = 16
    ):
        super(SENetBlock, self).__init__()
        
        # c_over_r = c_out // ratio
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        # self.global_avg_pool = GlobalAvgPooling()
        # self.se_fc1 = nn.Linear(c_out, c_over_r)
        # self.se_fc2 = nn.Linear(c_over_r, c_out)
        self.squeeze_excitation = SqueezeExcitation(c_out, ratio=ratio)
        self.shortcut = nn.Identity()
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(c_out),
            )
    
    # def _squeeze_excitation(self, x):
    #     scale = self.global_avg_pool(x)
    #     scale = scale.view(scale.size(0), -1)
    #     scale = F.relu(self.se_fc1(scale))
    #     scale = F.sigmoid(self.se_fc2(scale))
    #     scale = scale[:, :, None, None]
    #     x *= scale
    #     return x
            
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # x = self._squeeze_excitation(x)
        x = self.squeeze_excitation(x)
        x += identity
        out = F.relu(x)
        return out     
        

class SEBottleneck(nn.Module):
    expansion = 4
    def __init__(
      self,
      c_in,
      c_out,
      stride, 
      groups: int = 32,
      ratio = 16,
    ):
        
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(c_out)
        self.conv3 = nn.Conv2d(c_out, c_out * self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(c_out * self.expansion)
        self.squeeze_excitation = SqueezeExcitation(c_out * self.expansion, ratio=ratio)
        self.shortcut = nn.Identity()
        if c_in != c_out or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(c_in, c_out * self.expansion, kernel_size=1, stride=1),
                nn.BatchNorm2d(c_out * self.expansion)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = self.squeeze_excitation(x)
        x += identity
        out = F.relu(x)
        return out
        
class SENet(nn.Module):
    """
    Squeeze-and-Excitation Networks (Hu et al., CVPR 2018)
    paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Hu_Squeeze-and-Excitation_Networks_CVPR_2018_paper.pdf
    """
    def __init__(
        self,
        block,
        channels,
        num_blocks, 
        num_strides,
        c_init: int = 3, 
        groups: int = 32,
        num_classes: int = 1000,
    ):
        super(SENet, self).__init__()
        
        self.c_in = 64
        self.stem_conv = nn.Conv2d(c_init, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d((3,3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], num_strides[0])
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], num_strides[1])
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], num_strides[2])
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], num_strides[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1] * block.expansion, num_classes)
        
    def _make_layer(self, block, c_out, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride))
            self.c_in = c_out * block.expansion
        return layer
    
    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out 

def senet18(num_classes):
    return SENet(SENetBlock, [64, 128, 256, 512], [2, 2, 2, 2], [1, 2, 2, 2], num_classes=num_classes)  

def senet50(num_classes):
    return SENet(SENetBlock, [64, 128, 256, 512], [3, 4, 6, 3], [1, 2, 2, 2], num_classes=num_classes)  

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = SENet(SENetBlock, [64, 128, 256, 512], [3, 4, 6, 4], [1, 2, 2, 2], num_classes=10).to(device)
    x = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(x)
    print(output)
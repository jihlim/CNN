import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetV2Block(nn.Module):
    """
    Inverted Residual BLock
    """
    def __init__(
        self, 
        c_in, 
        c_out, 
        stride,
        padding = 1, 
        groups = 16,
        expansion = 6,
    ):
        super(MobileNetV2Block, self).__init__()
        
        c_mid = c_in * expansion
        self.stride = stride
        self.conv1 = nn.Conv2d(c_in, c_mid, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(c_mid)
        self.conv2 = nn.Conv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=c_mid)
        self.bn2 = nn.BatchNorm2d(c_mid)
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
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        out = self.bn3(self.conv3(x))
        if self.stride == 1:
            out += identity
        return out
    
    
class MobileNetV2(nn.Module):
    """
    MobileNetV2: Inverted Residuals and Linear Bottlenecks (Sandler et al. CVPR 2018)
    Paper: https://openaccess.thecvf.com/content_cvpr_2018/papers/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.pdf
    """
    def __init__(
      self,
      block,
      channels,
      num_blocks,
      c_init = 3, 
      groups = 1,
      num_classes = 1000,  
    ):
        super(MobileNetV2, self).__init__()
        
        self.c_in = 32
        self.stem_conv = nn.Conv2d(c_init, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], stride=1, expansion=1)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1], stride=1, expansion=6)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], stride=2, expansion=6)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], stride=2, expansion=6)
        self.layer5 = self._make_layer(block, channels[4], num_blocks[4], stride=1, expansion=6)
        self.layer6 = self._make_layer(block, channels[5], num_blocks[5], stride=2, expansion=6)
        self.layer7 = self._make_layer(block, channels[6], num_blocks[6], stride=1, expansion=6)
        self.final_conv = nn.Conv2d(channels[-1], 1280, kernel_size=1, stride=1)
        self.final_bn = nn.BatchNorm2d(1280)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1280, num_classes)
    
    def _make_layer(self, block, c_out, num_blocks, stride, expansion):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = nn.Sequential()
        for stride in strides:
            layer.append(block(self.c_in, c_out, stride, expansion)) 
            self.c_in = c_out  
        return layer
    
    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = F.relu(self.final_bn(self.final_conv(x)))
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        
        return out   
           
def mobilenetv2(num_classes):
    return MobileNetV2(MobileNetV2Block, [16, 24, 32, 64, 96, 160, 320], [1, 2, 3, 4, 3, 3, 1], num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = MobileNetV2(MobileNetV2Block, [16, 24, 32, 64, 96, 160, 320], [1, 2, 3, 4, 3, 3, 1], num_classes=10).to(device)
    input = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(input)
    print(output)
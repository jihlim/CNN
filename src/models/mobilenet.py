import torch
import torch.nn as nn
import torch.nn.functional as F


class MobileNetBlock(nn.Module):
    def __init__(
        self, 
        c_in, 
        c_out, 
        stride,
        padding = 1, 
        groups = 1,
    ):
        super(MobileNetBlock, self).__init__()
        
        # Depthwise Convolution
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=padding, groups=c_in)
        self.bn1 = nn.BatchNorm2d(c_in)
        # Pointwise Convolution
        self.conv2 = nn.Conv2d(c_in, c_out, kernel_size=1, stride=1)
        self.bn2 = nn.BatchNorm2d(c_out)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(x)))
        return out
    
    
class MobileNet(nn.Module):
    """
    MobileNets: Efficients Convolutional Neural Networks for Mobile Vision Applications
    Paper: https://arxiv.org/pdf/1704.04861
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
        super(MobileNet, self).__init__()
        
        self.c_in = 32
        self.stem_conv = nn.Conv2d(c_init, 32, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = block(channels[0], channels[1], stride=1)
        self.layer2 = self._make_layer(block, channels[1], channels[2], num_blocks[0], stride=1)
        self.layer3 = self._make_layer(block, channels[2], channels[3], num_blocks[1], stride=1)
        self.layer4 = self._make_layer(block, channels[3], channels[4], num_blocks[2], stride=2)
        self.layer5 = self._make_layer(block, channels[4], channels[5], num_blocks[3], stride=2)
        self.layer6 = block(channels[5], channels[5], stride=2, padding=4)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[-1], num_classes)
    
    def _make_layer(self, block, c_in, c_out, num_blocks, stride):
        layer = nn.Sequential()
        if num_blocks != 1:
            for _ in range(num_blocks-1):
                layer.append(block(c_in, c_in, stride=1))
        layer.append(block(c_in, c_out, stride=stride))    
        return layer
    
    def forward(self, x):
        x = F.relu(self.bn1(self.stem_conv(x)))
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
           
def mobilenet(num_classes):
    return MobileNet(MobileNetBlock, [32, 64, 128, 256, 512, 1024], [1, 2, 2, 6], num_classes=num_classes)
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    model = MobileNet(MobileNetBlock, [32, 64, 128, 256, 512, 1024], [1, 2, 2, 6], num_classes=10).to(device)
    input = torch.randn((1, 3, 32, 32), dtype=torch.float32, device=device)
    output = model(input)
    print(output)
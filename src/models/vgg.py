import torch
import torch.nn as nn
import torch.nn.functional as F 

class VGGBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
    ):
        super(VGGBlock, self).__init__()
        
        self.conv = nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = F.relu(self.conv(x))
        return out


class VGG(nn.Module):
    r"""
    Very Deep Convolutional Networks for Large-Scale Image Recognition (Simonyan and Zisserman, ICLR 2015)
    paper: https://arxiv.org/pdf/1409.1556
    """
    def __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        c_init: int = 3,
        num_classes: int = 1000,
        dropout_p: float = 0.5,
        add_lrn: bool = False,
        add_1x1: bool = False,
    ):
        super(VGG, self).__init__()

        self.c_in = c_init
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0], add_lrn=add_lrn)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1])
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2], add_1x1=add_1x1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3], add_1x1=add_1x1)
        self.max_pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.layer5 = self._make_layer(block, channels[4], num_blocks[4], add_1x1=add_1x1)
        self.max_pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(512 * 7 * 7, 4096)
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(4096, 4096)
        self.dropout2 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(4096, num_classes)

    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        add_lrn: bool = False,
        add_1x1: bool = False,
    ):
        
        layers = []
        for _ in range(num_blocks):
            layers.append(block(self.c_in, c_out))
            self.c_in = c_out
        
        if add_lrn:
            layers.append(self.lrn1)
        
        if add_1x1:
            layers.append(nn.Conv2d(self.c_in, c_out, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))
            self.c_in = c_out
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.max_pool1(x)
        x = self.layer2(x)
        x = self.max_pool2(x)
        x = self.layer3(x)
        x = self.max_pool3(x)
        x = self.layer4(x)
        x = self.max_pool4(x)
        x = self.layer5(x)
        x = self.max_pool5(x)

        x = x.view(-1, 512 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        out = self.fc3(x)
        return out

def vgg11(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [1, 1, 2, 2, 2], num_classes=num_classes)

def vgg11_lrn(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [1, 1, 2, 2, 2], num_classes=num_classes, add_lrn=True)

def vgg13(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [2, 2, 2, 2, 2], num_classes=num_classes)

def vgg16_c(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [2, 2, 2, 2, 2], num_classes=num_classes, add_1x1=True)

def vgg16(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [2, 2, 3, 3, 3], num_classes=num_classes)

def vgg19(num_classes):
    return VGG(VGGBlock, [64, 128, 256, 512, 512], [2, 2, 4, 4, 4], num_classes=num_classes)

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = vgg19(10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    out = model(x)
    print(out.shape)
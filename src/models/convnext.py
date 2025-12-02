import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNormalization2d(nn.Module):
    def __init__(
            self,
            d_out: int, 
            eps: float = 1e-5
    ):
        super(LayerNormalization2d, self).__init__()
        
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((1, d_out, 1, 1), dtype=torch.float32))
        self.beta = nn.Parameter(torch.zeros((1, d_out, 1, 1), dtype=torch.float32))

    def forward(self, x):
        mean = torch.mean(x, (1, 2, 3), keepdim=True)
        var = torch.var(x, (1, 2, 3), keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        x = x * self.gamma + self.beta
        return x


class ConvNeXtBlock(nn.Module):
    expansion: int = 4

    def __init__(
            self,
            c_in: int,
            stride: int = 1,
            padding: int = 1,
            groups: int = 32,
    ):
        super(ConvNeXtBlock, self).__init__()

        c_mid = c_in * self.expansion
        c_out = c_in
        self.conv1 = nn.Conv2d(c_in, c_in, kernel_size=7, stride=1, padding=3, groups=c_in)
        self.conv2 = nn.Conv2d(c_in, c_mid, kernel_size=1)
        self.conv3 = nn.Conv2d(c_mid, c_out, kernel_size=1)
        self.gelu = nn.GELU()

    def _layer_norm(self, x):
        b, c, h, w = x.shape
        x = F.layer_norm(x, [c, h, w], eps=1e-6)
        return x

    def forward(self, x):
        identity = x

        x = self.conv1(x)
        x = self._layer_norm(x)
        x = self.conv2(x)
        x = self.gelu(x)
        x = self.conv3(x)
        out = x + identity
        
        return out
        

class ConvNeXt(nn.Module):
    r"""
    A ConvNet for the 2020s (Liu et al., CVPR 2022)
    paper: https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf
    """
    def __init__(
        self, 
        block,
        channels: list,
        num_blocks: list,
        c_init: int = 3,
        num_classes: int = 1000,
    ):
        super(ConvNeXt, self).__init__()
        
        self.patchify_stem_conv = nn.Conv2d(c_init, channels[0], kernel_size=4, stride=4)
        self.layer1 = self._make_layer(block, channels[0], num_blocks[0])
        self.downsample1 = nn.Conv2d(channels[0], channels[1], kernel_size=2, stride=2)
        self.layer2 = self._make_layer(block, channels[1], num_blocks[1])
        self.downsample2 = nn.Conv2d(channels[1], channels[2], kernel_size=2, stride=2)
        self.layer3 = self._make_layer(block, channels[2], num_blocks[2])
        self.downsample3 = nn.Conv2d(channels[2], channels[3], kernel_size=2, stride=2)
        self.layer4 = self._make_layer(block, channels[3], num_blocks[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(channels[3], num_classes)

    def _make_layer(
        self,
        block, 
        c_in, 
        num_blocks
    ):
        
        layers = []
        for _ in range(num_blocks):
            layers.append(block(c_in))
        return nn.Sequential(*layers)

    def _layer_norm(self, x):
        b, c, h, w = x.shape
        x = F.layer_norm(x, [c, h, w], eps=1e-6)
        return x
    
    def forward(self, x):
        x = self.patchify_stem_conv(x)
        x = self._layer_norm(x)
        x = self.layer1(x)
        x = self._layer_norm(x)
        x = self.downsample1(x)
        x = self.layer2(x)
        x = self.downsample2(x)
        x = self._layer_norm(x)
        x = self.layer3(x)
        x = self._layer_norm(x)
        x = self.downsample3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = self._layer_norm(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out

def convnext_t(num_classes):
    return ConvNeXt(ConvNeXtBlock, [96, 192, 384, 768], [3, 3, 9, 3], num_classes=num_classes)

def convnext_s(num_classes):
    return ConvNeXt(ConvNeXtBlock, [96, 192, 384, 768], [3, 3, 27, 3], num_classes=num_classes)

def convnext_b(num_classes):
    return ConvNeXt(ConvNeXtBlock, [128, 256, 512, 1024], [3, 3, 27, 3], num_classes=num_classes)

def convnext_l(num_classes):
    return ConvNeXt(ConvNeXtBlock, [192, 384, 768, 1536], [3, 3, 27, 3], num_classes=num_classes)

def convnext_xl(num_classes):
    return ConvNeXt(ConvNeXtBlock, [256, 512, 1024, 2048], [3, 3, 27, 3], num_classes=num_classes)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNeXt(ConvNeXtBlock, [96, 192, 384, 768], [3, 3, 9, 3], num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    out = model(x)
    print(out.shape)
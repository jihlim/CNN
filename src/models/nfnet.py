import torch
import torch.nn as nn
import torch.nn.functional as F


# class AdaptiveGradientClipping(nn.Module):
#     def __init__(
#         self,
        
#     ):
#         super(AdaptiveGradientClipping, self).__init__()
        
#     def forward(self, x):
        

class NFNetSqueezeExcitation(nn.Module):
    def __init__(
        self,
        c_out,
        ratio: int = 2,
    ):
        super(NFNetSqueezeExcitation, self).__init__()
        
        c_over_r = int(c_out // ratio)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.se_fc1 = nn.Conv2d(c_out, c_over_r, kernel_size=1)
        self.se_fc2 = nn.Conv2d(c_over_r, c_out, kernel_size=1)
        
    def forward(self, x):
        scale = self.global_avg_pool(x)
        scale = F.relu(self.se_fc1(scale))
        scale = 2 * torch.sigmoid(self.se_fc2(scale))
        x = x * scale
        return x 


class WSConv2d(nn.Conv2d):
    r"""
    Pytorch Implementation of Weight Standardization Convolution
    
    - Useful Information
        pytorch Conv2d.weight = [output_channels, input_channels, kernel_size_h,  kernel_size_w]
        Haiku   Conv2d.weight = [kernel_size,     kernel_size,    input_channels, output_channels]
    """
    def __init__(
        self,
        c_in: int,
        c_out: int,
        kernel_size,
        stride = 1,
        padding = 0,
        dilation = 1, 
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device = None,
        dtype = None,
        eps: float = 1e-4,
    ): 
        super(WSConv2d, self).__init__(
            c_in,
            c_out,
            kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode = padding_mode,
            device = device,
            dtype = dtype,
        )
        self.eps = eps
        self.affine_gain = nn.Parameter(torch.ones((c_out, 1, 1, 1), dtype=torch.float32))
        self.affine_bias = nn.Parameter(torch.zeros((1, c_out, 1, 1), dtype=torch.float32)) 
    
    def forward(self, x):
        mean = self.weight.mean(dim=(1,2,3), keepdim=True)
        var = self.weight.var(dim=(1,2,3), keepdim=True)
        fan_in = torch.prod(torch.tensor(self.weight.shape[1:]), dim=0)
        weight = (self.weight - mean) / (((var * fan_in) ** 0.5) + self.eps)
        
        # Affine Gain
        weight = weight * self.affine_gain
        
        # Conv2d Forward
        out = F.conv2d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        
        # Affine Bias
        out = out + self.affine_bias
        return out


class NFNetBlock(nn.Module):
    def __init__(
        self,
        c_in: int,
        c_out: int,
        stride: int=1,
        groups: int = 2,
        expansion: int = 2, 
        ratio: int = 2,
        predicted_var: float = 1.0,
        nonlinearity: str = "gelu",
    ):
        super(NFNetBlock, self).__init__()
        
        nonlinearity_dict = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
        }
        gamma_dict = {
            "relu": 1.7139588594436646,
            "gelu": 1.7015043497085571,
        }
        self.c_in = c_in
        self.c_out = c_out
        self.alpha = 0.2
        self.one_over_beta = 1.0 / (predicted_var ** 0.5) 
        self.skipinit_gain = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        c_mid = int(c_out // expansion)
        self.conv1 = WSConv2d(c_in, c_mid, kernel_size=1)
        self.conv2 = WSConv2d(c_mid, c_mid, kernel_size=3, stride=stride, padding=1, groups=groups)
        self.conv3 = WSConv2d(c_mid, c_mid, kernel_size=3, stride=1, padding=1, groups=groups)
        self.conv4 = WSConv2d(c_mid, c_out, kernel_size=1)
        self.squeeze_excitation = NFNetSqueezeExcitation(c_out, ratio=ratio)
        self.activation = nonlinearity_dict[nonlinearity]
        self.gamma = gamma_dict[nonlinearity]
        self.shortcut = nn.Identity()
        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(2, stride=2),
                WSConv2d(c_in, c_out, kernel_size=1),
            )
        elif stride == 1 and c_in != c_out:
            self.shortcut = WSConv2d(c_in, c_out, kernel_size=1)
        self.stride = stride
        
    def forward(self, x):
        identity = x
        x = x * self.one_over_beta
        x = self.activation(x) * self.gamma
        if self.stride != 1 or self.c_in != self.c_out:
            identity = self.shortcut(x)
            
        x = self.activation(self.conv1(x)) * self.gamma
        x = self.activation(self.conv2(x)) * self.gamma
        x = self.activation(self.conv3(x)) * self.gamma
        x = self.conv4(x)
        x = self.squeeze_excitation(x)
        if self.stride == 1:
            x = x * self.skipinit_gain
        x = x * self.alpha
        out = x + identity
        return out        


class NFNet(nn.Module):
    r"""
    High-Performance Large-Scale Image Recognition Without Normalization (Brock et al., ICML 2021)
    paper: https://proceedings.mlr.press/v139/brock21a/brock21a.pdf
    
    - gamma:
        gamma (ReLU):   1.7139588594436646 
                        1.7128585504496627 = (2/(1-1/numpy.pi)) ** 0.5, numpy.pi = 3.141592653589793
        gamma (GELU):   1.7015043497085571
    """
    def __init__(
        self,
        block,
        channels: list,
        num_blocks: list,
        num_strides: list,
        c_init: int = 3, 
        num_classes: int = 1000,
        nonlinearity: str = "gelu",
        alpha: float = 0.2,
        predicted_var: float = 1.0,
        dropout_p: float = 0.2,
    ):
        super(NFNet, self).__init__()
        
        cardinality = [c//128 for c in channels[4:]]
        nonlinearity_dict = {
            "relu": nn.ReLU(inplace=True),
            "gelu": nn.GELU(),
        }
        gamma_dict = {
            "relu": 1.7139588594436646,
            "gelu": 1.7015043497085571,
        }
        self.c_in = channels[3]
        self.alpha = alpha
        self.nonlinearity = nonlinearity
        
        self.stem_conv1 = nn.Conv2d(c_init, channels[0], kernel_size=3, stride=2, padding=1)
        self.stem_conv2 = nn.Conv2d(channels[0], channels[1], kernel_size=3, stride=1, padding=1)
        self.stem_conv3 = nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1)
        self.stem_conv4 = nn.Conv2d(channels[2], channels[3], kernel_size=3, stride=1, padding=1)
        self.layer1 = self._make_layer(block, channels[4], num_blocks[0], num_strides[0], cardinality[0])
        self.layer2 = self._make_layer(block, channels[5], num_blocks[1], num_strides[1], cardinality[1])
        self.layer3 = self._make_layer(block, channels[6], num_blocks[2], num_strides[2], cardinality[2])
        self.layer4 = self._make_layer(block, channels[7], num_blocks[3], num_strides[3], cardinality[3])
        self.final_conv = nn.Conv2d(channels[-1], channels[-1] * 2, kernel_size=1)
        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout2d(p=dropout_p, inplace=True)
        self.activation = nonlinearity_dict[nonlinearity]
        self.gamma = gamma_dict[nonlinearity]
        self.fc = nn.Linear(channels[-1] * 2, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)
        self._initialization()
        
    def _initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
        
    def _make_layer(
        self,
        block,
        c_out: int,
        num_blocks: int,
        stride: int,
        groups: int,
        predicted_var: float = 1.0,
    ):
        strides = [stride] + [1] * (num_blocks - 1)
        layer = nn.Sequential()
        for stride in strides:
            # Variance Reset at Transition Block
            if stride != 1 or self.c_in != c_out:
                predicted_var = 1.0 
            layer.append(block(self.c_in, c_out, stride=stride, groups=groups, predicted_var=predicted_var, nonlinearity=self.nonlinearity))
            predicted_var = predicted_var + (self.alpha ** 2)
            self.c_in = c_out
        return layer 
    
    def forward(self, x): 
        x = self.activation(self.stem_conv1(x)) * self.gamma
        x = self.activation(self.stem_conv2(x)) * self.gamma
        x = self.activation(self.stem_conv3(x)) * self.gamma
        x = self.stem_conv4(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.final_conv(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        return out 


def nfnet_f0(num_classes):
    # 54 (12 * 4 + 6)Layers
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [1, 2, 6, 3], [1, 2, 2, 2], num_classes=num_classes)

def nfnet_f1(num_classes):
    # 102 Layers (24 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [2, 4, 12, 6], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.3)

def nfnet_f2(num_classes):
    # 150 Layers (36 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [3, 6, 18, 9], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.4)

def nfnet_f3(num_classes):
    # 198 Layers (48 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [4, 8, 24, 12], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.4)

def nfnet_f4(num_classes):
    # 246 Layers (60 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [5, 10, 30, 15], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.5)

def nfnet_f5(num_classes):
    # 294 Layers (72 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [6, 12, 36, 18], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.5)

def nfnet_f6(num_classes):
    # 342 Layers (84 * 4 + 6)
    return NFNet(NFNetBlock, [16, 32, 64, 128, 256, 512, 1536, 1536], [7, 14, 42, 21], [1, 2, 2, 2], num_classes=num_classes, dropout_p=0.5)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NFNet(NFNetBlock, [64, 256, 512, 1536, 1536],  [1, 2, 6, 3], [2, 2, 2, 2], num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    output = model(x)
    print(output.shape)
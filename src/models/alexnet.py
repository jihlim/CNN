import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    r"""
    ImageNet Classification with Deep Convolutional Neural Networks (Krizhevsky et al., NeurIPS 2012)
    paper: https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf
    """
    def __init__(
            self,
            c_init: int = 3,
            num_classes: int = 1000
    ):
        super(AlexNet, self).__init__()

        self.conv1 = nn.Conv2d(c_init, 96, kernel_size=11, stride=4, padding=2)
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(96, 256, kernel_size=5, padding=2)
        self.lrn2 = nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2)
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(256, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.fc1 = nn.Linear(256 * 6 * 6, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, num_classes)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.lrn1(x)
        x = self.max_pool1(x)
        
        x = F.relu(self.conv2(x))
        x = self.lrn2(x)
        x = self.max_pool2(x)

        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.max_pool3(x)

        x = x.view(-1, 256 * 6 * 6)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out

def alexnet(num_classes):
    return AlexNet(num_classes=num_classes)

if __name__ == '__main__':
    device =  torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet(num_classes=10).to(device)
    x = torch.randn((1, 3, 224, 224), dtype=torch.float32, device=device)
    out = model(x)
    print(out.shape)
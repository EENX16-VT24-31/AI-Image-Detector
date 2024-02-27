import torch.hub
from torch import nn
import torch.nn.functional as F

class FCN_test(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2)
        self.deconv2 = nn.ConvTranspose2d(32, 1, kernel_size=9, stride=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.deconv1(x))
        x = F.sigmoid(self.deconv2(x))
        return x

class FCN_resnet50(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', weights=None)
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, x):
        return F.sigmoid(self.model(x)["out"])


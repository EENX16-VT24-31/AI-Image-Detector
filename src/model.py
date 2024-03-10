from torch import nn
from torchvision import models  # type: ignore
import torch.nn.functional as F

# Model definition and model utilities


class BinaryResNet50NotPreTrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50: models.ResNet = models.resnet50(pretrained=False)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return F.sigmoid(self.resnet50(x))
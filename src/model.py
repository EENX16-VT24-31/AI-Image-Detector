from torch import nn
from torchvision import models  # type: ignore

# Model definition and model utilities


class BinaryResNet50PreTrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50: models.ResNet = models.resnet50(pretrained=True)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet50(x)

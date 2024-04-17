from torch import nn
from torchvision import models  # type: ignore

# Model definition and model utilities


class BinaryResNet50NotPreTrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50: models.ResNet = models.resnet50(pretrained=False)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x, apply_sigmoid=True):
        x = self.resnet50(x)
        if apply_sigmoid:
            x = self.activation(x)
        return x

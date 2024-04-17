import torch
from torch import nn
from torchvision import models  # type: ignore
from torch.functional import F


# Model definition and model utilities


class BinaryResNet50PreTrained(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50: models.ResNet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        num_features: int = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return F.sigmoid(self.resnet50(x))

    def load(self, weight_path):
        pretrained_data: dict[str:float] = torch.load(weight_path)
        weights: dict[str:float] = {key.replace("model.", ""): val for key, val in pretrained_data.items()}
        self.model.load_state_dict(weights)

import torch.hub
from torch import nn
import torch.nn.functional as F
from torchvision.models.segmentation import FCN

from src.FCN.config import MODEL_PATH

class _FCN_test(nn.Module):
    """
    Small model to illustrate the components of an FCN, not to be used!
    """
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
    def __init__(self, pretrained=False):
        super().__init__()
        self.model: FCN = torch.hub.load('pytorch/vision:v0.10.0', 'fcn_resnet50', weights="DEFAULT")
        self.model.classifier[4] = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        if pretrained:
            try:
                pretrained_data: dict = torch.load(MODEL_PATH)
            except FileNotFoundError:
                print("No pth file for the given model found, please check the path in config,"
                      " or train the model with the train.py script")
                exit(1337)
                return  # Not reachable, but PyCharm gets mad if it isn't here

            weights = {key.replace("model.", ""): val for key, val in pretrained_data.items()}
            self.model.load_state_dict(weights)

    def forward(self, x):
        return F.sigmoid(self.model(x)["out"])


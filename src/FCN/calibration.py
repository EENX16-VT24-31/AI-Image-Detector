import math

import torch
from torch import nn, optim
from tqdm import tqdm

from src.FCN.config import DATA_PATH
from src.FCN.model import FCN_resnet50
from src.data.gen_image import Datasets, Generator, DataLoader


def platt_scale(logits: torch.Tensor, a, b):
    def logistic(x):
        return 1 / (1 + math.e**(a * x + b))

    return logistic(logits)


def get_platt_params(model: nn.Module, val_loader: DataLoader, refinements=3):
    assert refinements >= 2, "The number of refinements needs to be at least 2"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    mse_criterion = nn.MSELoss().to(device)

    a = nn.Parameter(torch.ones(1, device=device) * -1)
    b = nn.Parameter(torch.ones(1, device=device))

    for i in range(refinements):
        print(f"Calibration cycle [{i+1}/{refinements}]")
        optimizer = optim.Adam([a, b], lr=10 ** -(i+1))
        for inputs, labels in tqdm(val_loader):
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
            optimizer.zero_grad()
            loss = mse_criterion(platt_scale(outputs, a, b), labels)
            loss.backward()
            optimizer.step()

    return a.item(), b.item()


if __name__ == "__main__":
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    datasets = Datasets(DATA_PATH, generators=[Generator.SD1_4])
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()
    a, b = get_platt_params(model, datasets.validation)
    print(a, b)

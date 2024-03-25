import math

import torch
from torch import nn, optim
from tqdm import tqdm

from src.FCN.config import DATA_PATH
from src.FCN.model import FCN_resnet50
from src.data.gen_image import Datasets, Generator, DataLoader


def platt_scale(logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Applies platt scaling with the given parameters to the given input
    :param logits: The input that is to be platt scaled
    :param params: Tensor of size (2,) containing the two parameters for platt scaling
    :return: The platt scaled input
    """
    assert params.shape == (2,), "params given to platt_scale are not of shape (2,)"

    # Define the logistic function, to be able to apply it on element basis to the input
    def logistic(x: torch.Tensor) -> torch.Tensor:
        return 1 / (1 + math.e ** (params[0] * x + params[1]))

    return logistic(logits)


def get_platt_params(model: nn.Module, val_loader: DataLoader, refinements: int = 3) -> torch.Tensor:
    """
    Calculate the two platt parameters for a given model and dataset
    :param model: The model that will be tested, its parameters will not change from this function call,
    but it might move device, and it will be set to eval mode
    :param val_loader: The dataset to use when finding the platt parameters, preferably, this should be the same
    as the validation set used when training the model
    :param refinements: The number of calibration cycles, learning rate starts at 1 and decreases by a factor 10 each
    cycle, this value needs to be at least 2 for reasonable results
    :return: A tensor with shape (2,) containing the two platt parameters
    """
    assert refinements >= 2, "The number of refinements needs to be at least 2"

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    mse_criterion: nn.MSELoss = nn.MSELoss().to(device)
    # -10, 5 was chosen as an arbitrary start point for the parameters
    platt_params: torch.Tensor = nn.Parameter(torch.tensor([-10.0, 5.0], device=device))

    for i in range(refinements):
        print(f"Calibration cycle [{i + 1}/{refinements}]")
        optimizer: optim.Optimizer = optim.Adam([platt_params], lr=10 ** -(i + 1))

        inputs: torch.Tensor
        labels: torch.Tensor
        for inputs, labels in tqdm(val_loader):
            # Get the data from the model and expand labels
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs: torch.Tensor = model(inputs)
                labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()

            # Fit platt parameters using model data
            optimizer.zero_grad()
            loss: torch.Tensor = mse_criterion(platt_scale(outputs, platt_params), labels)
            loss.backward()
            optimizer.step()

    return platt_params


# Example usage
if __name__ == "__main__":
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    datasets: Datasets = Datasets(DATA_PATH, generators=[Generator.SD1_4], image_count=100 * 32 * 10)
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()  # Will be called
    print(get_platt_params(model, datasets.validation))

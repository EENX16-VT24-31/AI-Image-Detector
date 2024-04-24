import math
import torch
from torch import nn, optim
from tqdm import tqdm
from src.VIT.config import  PLATT_PATH
from src.data.gen_image import DataLoader


def platt_scale(logits: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
    """
    Applies platt scaling with the given parameters to the given input
    :param logits: The input that is to be platt scaled
    :param params: Tensor of size (2,) containing the two parameters for platt scaling
    :return: The platt scaled input
    """
    assert params.shape == (2,), "params given to platt_scale are not of shape (2,)"

    logits = torch.softmax(logits.T, dim=0).T
    scaling_factor =  1 / (1 + math.e ** (params[0] * logits[:, 1] + params[1]))

    return scaling_factor


def get_platt_params(model: nn.Module | None = None, val_loader: DataLoader | None = None) -> torch.Tensor:
    """
    Calculate the two platt parameters for a given model and dataset
    :param model: The model that will be tested, its parameters will not change from this function call,
    but it might move device, and it will be set to eval mode
    :param val_loader: The dataset to use when finding the platt parameters, preferably, this should be the same
    as the validation set used when training the model
    :return: A tensor with shape (2,) containing the two platt parameters
    """
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # -10, 5 was chosen as an arbitrary start point for the parameters
    platt_params: torch.Tensor = nn.Parameter(torch.tensor([-10.0, 5.0], device=device))

    try:
        pretrained_data: torch.Tensor = torch.load(PLATT_PATH)
        print("Successfully loaded stored platt parameters")
        return pretrained_data
    except FileNotFoundError:
        print("No stored platt parameters for the given model, need to calibrate")

    assert model, "No stored platt params for given model name, model needed"
    assert val_loader, "No stored platt params for given model name, val_loader needed"

    model.to(device)
    model.eval()

    criterion = nn.MSELoss().to(device)

    refinements: int = 3
    for i in range(refinements):
        optimizer: optim.Optimizer = optim.Adam([platt_params], lr=100 ** -(i + 1))
        print(platt_params)

        inputs: torch.Tensor
        labels: torch.Tensor
        for batch, (inputs, labels) in enumerate(tqdm(val_loader, f"Calibrating, cycle [{i + 1}/{refinements}]")):
            # Get the data from the model and expand labels
            with torch.no_grad():
                inputs, labels = inputs.to(device), labels.to(device)
                outputs: torch.Tensor = model(inputs)

            # Fit platt parameters using model data
            optimizer.zero_grad()
            loss: torch.Tensor = criterion(platt_scale(outputs, platt_params), labels.float())
            loss.backward()
            optimizer.step()

    torch.save(platt_params, PLATT_PATH)
    return platt_params

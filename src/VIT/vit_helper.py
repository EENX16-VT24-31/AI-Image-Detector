from typing import Tuple
import torch
from tqdm import tqdm



def _train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               criterion: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: str) -> Tuple[float, float]:

    """
    Function that calculates the loss and acc for one epoch on the training data.

    Args:
        model (torch.nn.Module): The model to train.
        dataloader (torch.utils.data.DataLoader): The DataLoader to train with.
        criterion (torch.nn.Module): The function that calculates the loss and backpropagation.
        optimizer (torch.optim.Optimizer): The function that calculates the gradient.
        device (str): The device used by torch.

    Returns:
        train_loss (float): The train loss.
        train_acc (float): The train accuracy.
    """

    model.train()
    train_loss: float = 0
    train_acc: float= 0

    for batch, (X, y) in enumerate(tqdm(dataloader, "Training Network")):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = criterion(y_pred, y)
        train_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc


def _test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              criterion: torch.nn.Module,
              device: str) -> Tuple[float, float]:

    """
    Function that calculates the loss and acc for one epoch on the test data.

    Args:
        model (torch.nn.Module): The model to test.
        dataloader (torch.utils.data.DataLoader): The DataLoader to test with.
        criterion (torch.nn.Module): The function that calculates the loss.
        device (str): The device used by torch.

    Returns:
        test_loss (float): The Test loss.
        test_acc (float): The test accuracy.
    """

    model.eval()
    test_loss: float= 0
    test_acc: float= 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(tqdm(dataloader, "Evaluating Network")):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            loss = criterion(y_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_labels = y_pred_logits.argmax(dim=1)
            test_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module,
          train_loader: torch.utils.data.DataLoader,
          test_loader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          criterion: torch.nn.Module,
          epochs: int,
          device: str) -> None:

    """
    Function used to train the model and report the train and test loss for each epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_loader (torch.utils.data.DataLoader): The DataLoader for the training data.
        test_loader (torch.utils.data.DataLoader): The DataLoader for the testing data.
        optimizer (torch.optim.Optimizer): The function that calculates the gradient.
        criterion (torch.nn.Module): The function that calculates the loss and backpropagation.
        epochs (int): The number of epochs to train the model.
        device (str): the device used by torch.
    """

    model.to(device)
    print(f'Model: {model} is training.')

    for epoch in range(epochs):
        print(f'\nEpochs [Current/Total]: [{epoch+1}/{epochs}]')
        train_loss, train_acc = _train_step(model=model,
                                            dataloader=train_loader,
                                            criterion=criterion,
                                            optimizer=optimizer,
                                            device=device)

        test_loss, test_acc = _test_step(model=model,
                                         dataloader=test_loader,
                                         criterion=criterion,
                                         device=device)

        print(f'Train Loss => {train_loss:.4f}, ',
              f'Train Acc => {train_acc:.4f}, ',
              f'Eval Loss => {test_loss:.4f}, ',
              f'Eval Acc => {test_acc:.4f}')

    # # Save the trained model
    # save_model(epochs, model, optimizer, criterion, 'trained_model.pth')


def validate(model: torch.nn.Module,
             val_loader: torch.utils.data.DataLoader,
             criterion: torch.nn.Module,
             device: str,
             saved_model_path: str=""
             ) -> None:

    """
    Function to validate the model.

    Args:
        model (torch.nn.Module): The model to validate.
        val_loader (torch.utils.data.DataLoader): The DataLoader for the validation data.
        criterion (torch.nn.Module): The function that calculates the loss.
        saved_model_path (str): The path used to load a pretrained state dictionary to the model.
        device (str): The device to perform validation on
    """

    # Set the model to evaluation mode
    model.eval()

    # Load the testing data
    # Assuming test_loader provides batches of (inputs, targets)
    val_loss: float= 0
    val_acc: float= 0

    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(tqdm(val_loader, "Validating model")):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            loss = criterion(pred, targets)
            val_loss += loss.item()

            pred_labels = pred.argmax(dim=1)
            val_acc += (pred_labels == targets).sum().item()/len(pred_labels)
    # Calculate the average test loss
    val_loss = val_loss/len(val_loader) # type: ignore
    print(f"Val Loss: {val_loss:.4f}")

    # Calculate accuracy
    val_acc = val_acc/len(val_loader)
    print(f"Val Accuracy: {val_acc:.4f}")

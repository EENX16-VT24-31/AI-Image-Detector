import torch

def save_model(epochs : int, model, optimizer, criterion, output_path : str):
    """
    Function to save the trained model to disk.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, output_path)

def parameter_info(model, printing : bool=True):
    """
    Function that prints info about model.

    ---Returns---
    total_params : the total number of parameters of the model
    total_trainable_params : the total number of trainable parameters of the model
    """

    total_params : int = sum(p.numel() for p in model.parameters())

    total_trainable_params : int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if printing:
        print(model)
        print(f"{total_params:,} total parameters.")
        print(f"{total_trainable_params:,} training parameters.")

    return total_params, total_trainable_params

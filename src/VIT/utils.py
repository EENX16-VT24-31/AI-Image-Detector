import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import List, Tuple
from PIL import Image
import numpy as np
import torch.nn.functional as F

def save_model(epochs: int,
               model: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               criterion: torch.nn.Module,
               output_path : str) -> None:
    """
    Function to save the trained model to disk.

    Args:
        epochs (int): Number of epochs.
        model (torch.nn.Module): The model whose state_dict is saved.
        optimizer (torch.optim.Optimizer): The optimizer whose state_dict is saved.
        criterion (torch.nn.Module): The criterion that is saved.

        output_path (str): The path to the file that averything is saved.
    """
    torch.save({
                'epoch': epochs,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion': criterion,
                }, output_path)


def parameter_info(model: torch.nn.Module,
                   printing: bool=True) -> Tuple[int, int]:
    """
    Function that prints info about model.

    Args:
        model (torch.nn.Module): The model which information will be returned.
        printing (bool): boolean That decide if the parameters will be printed in the terminal. Defaults to True.

    Returns:
        total_params (int): The total number of parameters of the model.
        total_trainable_params (int): The total number of trainable parameters of the model.
    """

    total_params : int = sum(p.numel() for p in model.parameters())

    total_trainable_params : int = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if printing:
        print(model)
        print(f"{total_params:,} total parameters.")
        print(f"{total_trainable_params:,} training parameters.")

    return total_params, total_trainable_params


def pred_and_plot_image(
    model: torch.nn.Module,
    class_names: List[str],
    image_path: str,
    image_size: Tuple[int, int],
    device: str,
    transform: transforms.Compose=None) -> None:

    """
    Function that predicts the type of image and plots the result along with the image.

    Args:
        model (torch.nn.Module): The model that is used for prediction.
        class_names (List[str]): The class names which the model will predict.
        image_path (str): The path to the the image which is being predicted.
        image_size (Tuple[int, int]): The size used for resizing.
        transform (torchvision.transforms, optional): The transfrom used on the image.
                                                      If None then a standard transform is used.
        device (str): The device used for computation.
    """


    image = Image.open(image_path)

    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

    model.to(device)

    # Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():

        transformed_image = image_transform(image).unsqueeze(dim=0)
        target_image_pred = model(transformed_image.to(device))


    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    plt.figure()
    plt.imshow(image)
    plt.title(
        f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}"
    )
    plt.axis(False)
    plt.show()


def heatmap(image_path: str,
            model: torch.nn.Module,
            device: str,
            image_size: Tuple[int, int] = (224, 224),
            patch_size: int = 16) -> None:
    """
    Creates an image with an attention map overlayed on top of the original image.

    Args:
        image_path (str): The path to the image we want to make a heatmap with.
        model (torch.nn.Module): The model whose attention we want to use.
        device (str): The device that handles the torch operations.
        image_size (Tuple[int, int]): The size of the resulting image and heatmap.
        patch_size (int): The patch size used when calculating the resulting heatmap.
    """

    # Open and transform the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image)

    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Extract patches from the image
    patches = model.patch_embedding(image_tensor.unsqueeze(0))

    # Pass patches through the input layer and attach class token and position embedding
    patches = model.input_layer(patches.float())
    transformer_input = torch.cat((model.cls_token, patches), dim=1) + model.pos_embedding

    # Pass input through the first linear layer in the transformer
    transformer_input_expanded = model.transformer[0].mlp[0](transformer_input)

    # Split querry, key, value into multiple q, k, and v vectors for multi-head attention
    qkv = transformer_input_expanded
    qkv = qkv.reshape(model.num_patches + 1, 3, 16, 64)
    q = qkv[:, 0].permute(1, 0, 2)
    k = qkv[:, 1].permute(1, 0, 2)
    kT = k.permute(0, 2, 1)

    # Calculate attention matrix
    attention_matrix = q @ kT

    # Average the attention weights across all heads
    attention_matrix_mean = torch.mean(attention_matrix, dim=0)

    # Add an identity matrix to account for residual connections
    residual_att = torch.eye(attention_matrix_mean.size(1)).to(device)
    aug_att_mat = attention_matrix_mean + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Resize attention map
    attn_heatmap = joint_attentions[0, 1:].reshape((int(image_size[0] / patch_size), int(image_size[1] / patch_size)))
    attn_heatmap_resized = F.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0),
                                         [image_size[0], image_size[1]],
                                         mode='bilinear').view(image_size[0], image_size[1], 1)

    # Create a plot for the image with the attention map overlayed
    fig, ax = plt.subplots(figsize=(5, 5))

    # Display the original sample with reduced opacity
    image = np.asarray(image_tensor.cpu()).transpose(1, 2, 0)
    ax.imshow(image, alpha=0.6)

    # Display the attention map with reduced opacity
    attn_heatmap = attn_heatmap_resized.detach().cpu().numpy().squeeze()
    ax.imshow(attn_heatmap, cmap='jet', alpha=0.4)
    ax.set_title("Attention Heatmap")

    # Turn off axis labels and ticks
    ax.set_axis_off()

    # Show the plot
    plt.show()

def set_seeds(seed: int=42) -> None:
    """
    Sets random seed for torch operations.

    Args:
        seed (int, optional): Random seed to set. Default: 42.
    """
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)

def heatmap_b16(image_path: str,
            model: torch.nn.Module,
            device: str,
            image_size: Tuple[int, int] = (224, 224),
            patch_size: int = 16) -> None:
    """
    Creates an image with an attention map overlayed on top of the original image.

    Args:
        image_path (str): The path to the image we want to make a heatmap with.
        model (torch.nn.Module): The model whose attention we want to use.
        device (str): The device that handles the torch operations.
        image_size (Tuple[int, int]): The size of the resulting image and heatmap.
        patch_size (int): The patch size used when calculating the resulting heatmap.
    """

    # Open and transform the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).to(device)

    # Move model to the specified device and set to evaluation mode
    model.to(device)
    model.eval()

    # Extract patches from the image
    patches = model.model._process_input(image_tensor.unsqueeze(0))

    # Attach class token and position embedding to patches

    transformer_input = torch.cat((model.model.class_token, patches), dim=1) + model.model.encoder.pos_embedding

    # Pass input through the first linear layer in the transformer
    transformer_input_expanded = model.model.encoder.layers.encoder_layer_0.mlp[0](transformer_input)

    # Split querry, key, value into multiple q, k, and v vectors for multi-head attention
    qkv = transformer_input_expanded
    qkv = qkv.reshape(model.model.seq_length, 4, 16, 64)
    q = qkv[:, 0].permute(1, 0, 2)
    k = qkv[:, 1].permute(1, 0, 2)
    kT = k.permute(0, 2, 1)

    # Calculate attention matrix
    attention_matrix = q @ kT

    # Average the attention weights across all heads
    attention_matrix_mean = torch.mean(attention_matrix, dim=0)

    # Add an identity matrix to account for residual connections
    residual_att = torch.eye(attention_matrix_mean.size(1)).to(device)
    aug_att_mat = attention_matrix_mean + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size()).to(device)
    joint_attentions[0] = aug_att_mat[0]
    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Resize attention map
    attn_heatmap = joint_attentions[0, 1:].reshape((int(image_size[0] / patch_size), int(image_size[1] / patch_size)))
    attn_heatmap_resized = F.interpolate(attn_heatmap.unsqueeze(0).unsqueeze(0),
                                         [image_size[0], image_size[1]],
                                         mode='bilinear').view(image_size[0], image_size[1], 1)

    # Create a plot for the image with the attention map overlayed
    fig, ax = plt.subplots(figsize=(5, 5))

    # Display the original sample with reduced opacity
    image = np.asarray(image_tensor.cpu()).transpose(1, 2, 0)
    ax.imshow(image, alpha=0.6)

    # Display the attention map with reduced opacity
    attn_heatmap = attn_heatmap_resized.detach().cpu().numpy().squeeze()
    ax.imshow(attn_heatmap, cmap='jet', alpha=0.4)
    ax.set_title("Attention Heatmap")

    # Turn off axis labels and ticks
    ax.set_axis_off()

    # Show the plot
    plt.show()

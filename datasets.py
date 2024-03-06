import os
from typing import List, Tuple
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def load_images_data(train_path, val_path, transform, train_split=0.8, test_split=0.2, batch_size=32, size1=None, size2=None, shuffle=True):
    """
    Load image data from a folder and create DataLoader for train and test data.

    Args:
        path (str): Path to the folder containing image data.
        train_split (float): Percentage of data to be used for training (default: 0.8).
        test_split (float): Percentage of data to be used for testing (default: 0.2).
        batch_size (int): Batch size for DataLoader (default: 32).
        shuffle (bool): Whether to shuffle the data (default: True).

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
    """

    assert train_split + test_split == 1, "The sum of train_split and test_split must equal 1"
    assert size1 % batch_size == 0, "The size1 must be divisable with the batch size"
    assert size2 % batch_size == 0, "The size2 must be divisable with the batch size"


    # Load dataset from the image folder
    dataset = ImageFolder(root=train_path, transform=transform)
    val = ImageFolder(root=val_path, transform=transform)
    classes = dataset.classes 

    if size1:
        dataset = torch.utils.data.Subset(dataset, np.random.choice(len(dataset), size1, replace=False))
    if size2:
        val = torch.utils.data.Subset(val, np.random.choice(len(val), size2, replace=False))

    # Calculate sizes for train and test datasets
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size

    assert train_size % batch_size == 0, "The Train size must be divisable with the batch size"
    assert test_size % batch_size == 0, "The Test size must be divisable with the batch size"

    # Split dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'Length of train data: {len(train_dataset)}')
    print(f'Length of validation data: {len(val)}')

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)  # No need to shuffle test data
    val_loader =DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True)  # No need to shuffle validation data

    return train_loader, test_loader, val_loader, classes





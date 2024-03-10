from typing import List, Tuple
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split

def create_loaders(train_path: str,
                     val_path:str,
                     transform: transforms.Compose,
                     train_split: float=0.8,
                     test_split: float=0.2,
                     batch_size: int=32,
                     size1: int=0,
                     size2: int=0,
                     shuffle: bool=True) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """
    Create DataLoaders for train, test and val data.

    Args:
        train_path (str): Path to the folder containing the training images.
        val_path (str): Path to the folder containing the validation images.
        transform (transforms.Compose): The transformer used to create the image tensors.
        train_split (float, optional): Percentage of data to be used for training. Default: 0.8.
        test_split (float, optional): Percentage of data to be used for testing. Default: 0.2.
        batch_size (int, optional): Batch size for DataLoader. Default: 32.
        size1 (int, optional): The portion of the traning folder to use if not all images are wanted. Default: None.
        size2 (int, optional): The portion of the validation folder to use if not all images are wanted. Default: None.
        shuffle (bool, optional): Whether to shuffle the data. Default: True.

    Returns:
        train_loader (DataLoader): DataLoader for training data.
        test_loader (DataLoader): DataLoader for testing data.
        val_loader (DataLoader): DataLoader for validation data.
        classes (List[str]): List of the names of the classes.
    """

    assert train_split + test_split == 1, "The sum of train_split and test_split must equal 1"

    # Load dataset from the image folder
    dataset = ImageFolder(root=train_path, transform=transform)
    val = ImageFolder(root=val_path, transform=transform)
    classes = dataset.classes

    if size1 > 0:
        indices1 = np.random.choice(len(dataset), size1, replace=False)
        dataset = torch.utils.data.Subset(dataset, indices1)
    if size2 > 0:
        indices2 = np.random.choice(len(val), size2, replace=False)
        val = torch.utils.data.Subset(val, indices2)

    # Calculate sizes for train and test datasets
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    test_size = total_size - train_size

    # Split dataset into train and test sets
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    print(f'Length of train data: {len(train_dataset)}')
    print(f'Length of validation data: {len(val)}')

    # Create DataLoader for train and test datasets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    val_loader =DataLoader(val, batch_size=batch_size, shuffle=False, pin_memory=True)

    return train_loader, test_loader, val_loader, classes





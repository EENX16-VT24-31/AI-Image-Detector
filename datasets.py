import os
from typing import List, Tuple
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader




def create_dataloaders(train_path: str,
                       test_path: str,
                       val_path: str,
                       transform: transforms.Compose,
                       batch_size: int,
                       num_workers: int) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:

    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    path_to_test_images: str = os.path.join(dir_path, test_path)
    path_to_validate_images: str = os.path.join(dir_path, val_path)
    path_to_train_images: str = os.path.join(dir_path, train_path)
   
    # Creating the datasets
    train_data = datasets.ImageFolder(root=path_to_train_images, transform=transform)
    validate_data = datasets.ImageFolder(root=path_to_validate_images, transform=transform)
    test_data = datasets.ImageFolder(path_to_test_images, transform=transform)
    
    class_names = train_data.classes

    # training data loaders
    train_loader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )

    # validate data loaders
    val_loader = DataLoader(
        validate_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    # testing data loaders
    test_loader = DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, class_names


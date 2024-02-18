import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader


dir_path: str = os.path.dirname(os.path.realpath(__file__))
path_to_test_images: str = os.path.join(dir_path, r'data\Dataset-Mini\test')
path_to_train_images: str = os.path.join(dir_path, r'data\Dataset-Mini\train')

# batch size
BATCH_SIZE = 64

# the training transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
    transforms.RandomRotation(degrees=(30, 70)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# the testing transforms
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.5, 0.5, 0.5],
        std=[0.5, 0.5, 0.5]
    )
])

# training dataset
train_dataset = datasets.ImageFolder(
    root=path_to_train_images, 
    transform=train_transform
    )

# test dataset
test_dataset = datasets.ImageFolder(
    path_to_test_images, 
    transform=test_transform
    )


# training data loaders
train_loader = DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True
)

# testing data loaders
test_loader = DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True
)


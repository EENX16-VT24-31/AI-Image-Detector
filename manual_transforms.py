from typing import Tuple
import torchvision.transforms as transforms

def create_transform(image_size: Tuple[int, int]) -> transforms.Compose:

    """
    Function that creates a transformation function

    Args:
        image_size (Tuple[int, int]):  The size used to resize the image.

    Retruns:
        transform (transforms.Compose): A torchvision transform.
    """

    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return transform

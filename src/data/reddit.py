from torch.utils.data import Subset
from torchvision.datasets import ImageFolder
from torchvision import transforms

VALIDATION_PERCENTAGE = 0.1


class Datasets:
    def __init__(self, path: str, transform: transforms.Compose | None = None):
        if not transform:
            transform: transforms.Compose = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        image_folder: ImageFolder
        try:
            image_folder = ImageFolder(path, transform=transform)
        except FileNotFoundError:
            print("The reddit image folder was not found at", path)
            exit(1337)
            return

        n: int = len(image_folder)
        self.validation = Subset(image_folder, range(int(n * VALIDATION_PERCENTAGE)))
        self.testing = Subset(image_folder, range(int(n * VALIDATION_PERCENTAGE), n))

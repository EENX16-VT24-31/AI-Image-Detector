import torch
from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms, datasets

from enum import Enum
import os.path

from src.data.config import IMAGE_SIZE


class Generator(Enum):
    """
    Enum representing possible AI image generators, if the entire dataset is needed, use Generator.ALL.
    """
    ALL = 0
    SD1_4 = 1
    SD1_5 = 2
    MIDJOURNEY = 3
    WUKONG = 4
    VQDM = 5
    BIGGAN = 6
    GLIDE = 7
    ADM = 8


class Datasets:
    """
    Load the GenImage dataset, either in full, or partially, given a list of the desired Generators needed
    (Default ALL).
    Requires the path to the base folder "GenImage", and then randomly splits the training data into training and
    validationm, as well as loads the test_data.

    base_path:param required, path to GenImage folder
    split:param default = (0.9, 0.1), split for training and validation data
    batch_size:param default = 32, number of pictures loaded in each batch
    num_workers:param default = os.cpu_count(), number of workers per dataset
    rgb_mean:param default = (0.5, 0.5, 0.5), mean values per rgb color
    rgb_std:param default = (0.5, 0.5, 0.5), standard deviation values per rgb color
    generators:param list of generators to use, if undefined set to None, and all generators will be used
    transform:param transforms to apply to images, if undefined, a standard transform will be applied
    image_count:param limit the number of images in training + validation set by setting image_count to a smaller value
    """

    def __init__(self, base_path: str, split: tuple[float, float] = (0.9, 0.1),
                 batch_size: int = 32, num_workers: int | None = os.cpu_count(),
                 rgb_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
                 rgb_std: tuple[float, float, float] = (0.5, 0.5, 0.5),
                 generators: list[Generator] | None = None, transform: transforms.Compose | None = None,
                 image_count: int | None = None):

        if generators is None:
            generators = [Generator.ALL]

        assert sum(split) == 1 and len(split) == 2, "The split parameter should be a list with 2 parameters, summing " \
                                                    "to 1 "

        assert os.path.isdir(base_path), "The path provided should be the parent directory"

        assert len(rgb_mean) == len(rgb_std) == 3, "Provide means and standard deviations as tuples of length 3"
        assert all([0 <= x <= 1 for x in rgb_mean + rgb_std]), "All mean and standard deviation values must be" \
                                                               "between 0 and 1"

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers if num_workers else 4
        self.image_size: int = IMAGE_SIZE
        self.split: tuple[float, float] = split
        self.rgb_mean: tuple[float, float, float] = rgb_mean
        self.rgb_std: tuple[float, float, float] = rgb_std

        if transform is None:
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomCrop(self.image_size, pad_if_needed=True),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.Normalize(self.rgb_mean, self.rgb_std)
            ])

        sub_dirs: list[str] = [os.path.join(base_path, subdir) for subdir in os.listdir(base_path) if
                               os.path.isdir(os.path.join(base_path, subdir))]

        if Generator.ALL not in generators:
            generator_path_names: dict[Generator, str] = {
                Generator.SD1_4: "sdv4",
                Generator.SD1_5: "sdv5",
                Generator.MIDJOURNEY: "midjourney",
                Generator.WUKONG: "wukong",
                Generator.VQDM: "vqdm",
                Generator.BIGGAN: "biggan",
                Generator.GLIDE: "glide",
                Generator.ADM: "adm"
            }
            # Remove all unwanted image generators
            sub_dirs = list(filter(
                lambda path: any([generator_path_names[generator] in path for generator in generators]
                                 ), sub_dirs))

        training_images: ConcatDataset = \
            ConcatDataset([datasets.ImageFolder(os.path.join(path, "train"), transform=transform) for path in sub_dirs])

        test_images: ConcatDataset = \
            ConcatDataset([datasets.ImageFolder(os.path.join(path, "val"), transform=transform) for path in sub_dirs])

        self.image_count: int
        if image_count and image_count < len(training_images):
            self.image_count = image_count
        else:
            self.image_count = len(training_images)

        train_size: int = int(self.split[0] * self.image_count)
        val_size: int = self.image_count - train_size
        trash_size: int = len(training_images) - train_size - val_size

        train_set: Subset
        val_set: Subset
        train_set, val_set, _ = \
            random_split(training_images, [train_size, val_size, trash_size], generator=torch.manual_seed(1337))

        self.training: DataLoader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
        self.validation: DataLoader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
        self.testing: DataLoader = DataLoader(test_images, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
        self.classes: list[str] = ['ai', 'nature']


# Example Usage
if __name__ == "__main__":
    base_path: str = "E://GenImage"
    # dataset: Datasets = Datasets(base_path, generators=[Generator.ALL])
    dataset: Datasets = Datasets(base_path, generators=[Generator.SD1_4, Generator.MIDJOURNEY])

    i: int
    images: torch.Tensor
    labels: torch.Tensor

    print("Training Labels:")
    for i, (images, labels) in enumerate(dataset.training):
        print(labels)
        if i == 10:
            break

    print("Validation Labels:")
    for i, (images, labels) in enumerate(dataset.validation):
        print(labels)
        if i == 10:
            break

    print("Testing Labels:")
    for i, (images, labels) in enumerate(dataset.testing):
        print(labels)
        if i == 10:
            break

from torch.utils.data import DataLoader, random_split, ConcatDataset, Subset
from torchvision import transforms, datasets

import os.path
import matplotlib.pyplot as plt


class Datasets:
    """
    Load the UniversalFakeDetect dataset, given the path to the base folder "progan_train", and split it randomly into a
    training, validation and testing dataset, weighted according to the split argument. The transform for the images is
    a random crop of size 224x224, random flips horizontally and vertically, as well as normalization.
    """

    def __init__(self, base_path: str, split: tuple[float, float, float],
                 batch_size: int = 32, num_workers: int | None = 1, # os.cpu_count(), #0
                 rgb_mean: tuple[float, float, float] = (0.5, 0.5, 0.5),
                 rgb_std: tuple[float, float, float] = (0.5, 0.5, 0.5)):
        assert sum(split) == 1 and len(split) == 3, "The split parameter should be a list with 3 parameters, summing " \
                                                    "to 1 "

        assert os.path.isdir(base_path), "The path provided should be the parent directory"

        assert len(rgb_mean) == len(rgb_std) == 3, "Provide means and standard deviations as tuples of length 3"
        assert all([0 <= x <= 1 for x in rgb_mean + rgb_std]), "All mean and standard deviation values must be" \
                                                               "between 0 and 1"

        self.batch_size: int = batch_size
        self.num_workers: int = num_workers if num_workers else 1#0
        self.image_size: int = 224  # Should maybe be pulled from some global variable instead
        self.split: tuple[float, float, float] = split
        self.rgb_mean: tuple[float, float, float] = rgb_mean
        self.rgb_std: tuple[float, float, float] = rgb_std

        transform: transforms.Compose = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop(self.image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Normalize(self.rgb_mean, self.rgb_std)
        ])

        sub_dirs: list[str] = [os.path.join(base_path, subdir) for subdir in os.listdir(base_path) if
                               os.path.isdir(os.path.join(base_path, subdir))]

        images: ConcatDataset = ConcatDataset([datasets.ImageFolder(path, transform=transform) for path in sub_dirs])

        self.image_count: int = len(images)

        train_size: int = int(self.split[0] * self.image_count)
        val_size: int = int(self.split[1] * self.image_count)
        test_size: int = self.image_count - train_size - val_size

        self.train_set: Subset
        self.val_set: Subset
        self.test_set: Subset
        self.train_set, self.val_set, self.test_set = random_split(images, [train_size, val_size, test_size])

        self.training: DataLoader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                                               num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
        self.validation: DataLoader = DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                                                 num_workers=self.num_workers, persistent_workers=True, pin_memory=True)
        self.testing: DataLoader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                                              num_workers=self.num_workers, persistent_workers=True, pin_memory=True)


# Example Usage
# if __name__ == "__main__":
#     base_path: str = "/media/erwinia/T9/progan_train"
#     split: tuple[float, float, float] = (0.8, 0.1, 0.1)
#     dataset: Datasets = Datasets(base_path, split)

#     i: int
#     images: torch.Tensor
#     labels: torch.Tensor

#     print("Training Labels:")
#     for i, (images, labels) in enumerate(dataset.training):
#         print(labels)
#         if i == 10:
#             break

#     print("Validation Labels:")
#     for i, (images, labels) in enumerate(dataset.validation):
#         print(labels)
#         if i == 10:
#             break

#     print("Testing Labels:")
#     for i, (images, labels) in enumerate(dataset.testing):
#         print(labels)
#         if i == 10:
#             break

#Plot some images from the dataset
def plot_images(set, num_images=10):
    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for j, (images, labels) in enumerate(set):
        for i in range(num_images):
            image, label = images[i,:,:,:], labels[i]
            axes[i].imshow(image.permute(1, 2, 0))  # Permute the dimensions for correct display
            axes[i].set_title("Label: %d" % label)
            axes[i].axis('off')
        break

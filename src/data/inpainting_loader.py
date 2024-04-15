import PIL.ImageOps
import torch
import torchvision.transforms
from PIL import Image

import os
import math
import random


def pil_loader(path: str, mask: bool = False) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img: Image.Image = Image.open(f)
        if mask:
            img = PIL.ImageOps.invert(img)
            return img.convert("1")  # Black and white
        return img.convert("RGB")


class InpaintingDataset:
    """
    Class to load the inpainting dataset, very inefficient code, but its fine due to the limited number of images
    """

    def __init__(self, root_dir: str, data_set: str, transform: torchvision.transforms.Compose = None):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        assert data_set.lower() in ["training", "validation", "testing", "train", "val", "test"], \
            "data_set parameter must be set to training/train, validation/val or testing/test"

        self.root_dir: str = root_dir
        assert os.path.isdir(self.root_dir), "Root dir doesnt exist"

        self.image_folder: str = os.path.join(root_dir, "images")
        assert os.path.isdir(self.image_folder), "No images folder in root folder"

        self.mask_folder: str = os.path.join(root_dir, "labels")
        assert os.path.isdir(self.mask_folder), "No labels folder in root folder"

        assert len(os.listdir(self.image_folder)) == len(
            os.listdir(self.mask_folder)), "Differing length of input and labels folder"

        self.transform: torchvision.transforms.Compose = transform

        self.low_cut: float
        self.high_cut: float
        if data_set.lower() in ["training", "train"]:
            self.low_cut = 0
            self.high_cut = 0.8
        elif data_set.lower() in ["validation", "val"]:
            self.low_cut = 0.8
            self.high_cut = 0.9
        else:
            self.low_cut = 0.9
            self.high_cut = 1

        self.low_cut_idx: int = math.floor(self.low_cut * len(os.listdir(self.image_folder)))
        self.high_cut_idx: int = math.floor(self.high_cut * len(os.listdir(self.image_folder)))

    def __len__(self) -> int:
        return self.high_cut_idx - self.low_cut_idx

    def __iter__(self):
        images: list[str] = os.listdir(self.image_folder)
        masks: list[str] = os.listdir(self.mask_folder)

        data: list[tuple[str, str]] = list(zip(images, masks))
        random.Random(420).shuffle(data)

        data = data[self.low_cut_idx:self.high_cut_idx]
        to_tensor = torchvision.transforms.ToTensor()

        image: str
        mask: str
        for image, mask in data:
            image_path: str = os.path.join(self.image_folder, image)
            mask_path: str = os.path.join(self.mask_folder, mask)

            image_pil: Image.Image = pil_loader(image_path)
            mask_pil: Image.Image = pil_loader(mask_path, True).resize(image_pil.size)

            if self.transform:
                yield self.transform(image_pil)[None], to_tensor(mask_pil)[None]
            else:
                yield image_pil, mask_pil


# Example Usage
if __name__ == "__main__":
    root: str = r"/media/erwinia/T9/inpainting"
    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    inpainting_training: InpaintingDataset = InpaintingDataset(root, "training", transform)
    inpainting_validation: InpaintingDataset = InpaintingDataset(root, "val", transform)
    inpainting_testing: InpaintingDataset = InpaintingDataset(root, "testing", transform)

    i: int
    image: torch.Tensor
    label: torch.Tensor
    for i, (image, label) in enumerate(inpainting_testing):
        print(image, label)

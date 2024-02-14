from PIL import Image
import PIL
from pathlib import Path
import os
from data import universal_fake_detect
import torch

class ImageStatistics:

    def __init__(self):
        self.R_sum: int = 0
        self.G_sum: int = 0
        self.B_sum: int = 0
        self.R_MEAN: float = 0
        self.G_MEAN: float = 0
        self.B_MEAN: float = 0
        self.pixel_count: int = 0
        self.R_deviation: float = 0
        self.G_deviation: float = 0
        self.B_deviation: float = 0


    def sum_image_data(self, image: Image.Image):
        pixels: PIL.PyAccess = image.load()

        r: int
        g: int
        b: int
        r, g, b = pixels.getPixel[1,1]
        self.pixel_count += 1
        self.R_sum += r
        self.G_sum += g
        self.B_sum += b

    
    def calculate_mean(self) -> tuple[float, float, float]:     #tuple(float, float, float) gav fel, tuple[float, float, float] gjorde inte det
        R_mean: float = self.R_sum / self.pixel_count
        G_mean: float = self.G_sum / self.pixel_count
        B_mean: float = self.B_sum / self.pixel_count
        return (R_mean, G_mean, B_mean)


    def calculate_deviation(self, image: Image.Image):
        pixels: PIL.PyAccess = image.load()

        r: int
        g: int
        b: int

        r, g, b = pixels.getPixel[1,1]
        self.R_deviation += (r - self.R_MEAN) ** 2
        self.G_deviation += (g - self.G_MEAN) ** 2
        self.B_deviation += (b - self.B_MEAN) ** 2
        pass

    def calculate_std(self) -> tuple[float, float, float]: #tuple(float, float, float) gav fel, tuple[float, float, float] gjorde inte det
        R_std: float = (self.R_deviation / self.pixel_count)
        G_std: float = (self.G_deviation / self.pixel_count)
        B_std: float = (self.B_deviation / self.pixel_count)
        return (R_std, G_std, B_std)


if __name__ == "__main__":

    base_path: str = "/media/erwinia/T9/progan_train" # This is a correct path for erwinia but not for me
    split: tuple[float, float, float] = (1, 0, 0)
    dataset: universal_fake_detect.Datasets = universal_fake_detect.Datasets(base_path, split)

    i: int
    image: torch.Tensor
    label: torch.Tensor

    print("Training Labels:")
    for i, (image, label) in enumerate(dataset.training):
        image = image / 2 + 0.5 #unnormalize
        print("wertyuiop")
        ImageStatistics.sum_image_data(ImageStatistics.self, image)
        if i == 10:
            break

    ImageStatistics.calculate_mean(ImageStatistics.self)

    for i, (image, label) in enumerate(dataset.training):
        image = image / 2 + 0.5 #unnormalize
        print("wertyuiop")
        ImageStatistics.calculate_deviation(ImageStatistics.self, image)
        if i == 10:
            break
    
    ImageStatistics.calculate_std(ImageStatistics.self)
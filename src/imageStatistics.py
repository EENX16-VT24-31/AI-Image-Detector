from PIL import Image
from pathlib import Path
import os
import ImageLoader

class ImageStatistics:
    RGB_sum: int

    def __init__(self, path: os.PathLike[str]):
        self.path: os.PathLike[str] = path
        

    def calculate_image_data(image: Image.Image):
        pixels: PIL.PyAccess = image.load()
        r: int
        g: int
        b: int
        r, g, b = pixels.getPixel[1,1]


    def calculate_statistics():
        pass


if __name__ == "__main__":
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    path_to_images: str = os.path.join(dir_path, "Dataset-Mini")
    image_loader = ImageLoader(Path(path_to_images))

    # Iterate through the images using the generator
    img: Image.Image
    for img in image_loader:
        calculate_image_data(img)
    calculate_statistics()
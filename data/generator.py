from typing import Generator, Any

from PIL import Image

from pathlib import Path
import os


class ImageLoader:
    """
    Initialize a ImageLoader with the root path, the generator will find all images in all subdirectories of that path.
    """

    def __init__(self, path: os.PathLike[str]):
        self.path: os.PathLike[str] = path

    """
    Convert a full path to a pillow Image
    """

    @staticmethod
    def load_image(image_path: Path) -> Image.Image | None:
        # Load an image using PIL
        try:
            image: Image.Image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def __iter__(self) -> Generator[Image.Image, Any, None]:
        for root, dirs, files in os.walk(self.path):
            file: str
            for file in files:
                file_path: str = os.path.join(root, file)
                image: Image.Image | None = self.load_image(Path(file_path))
                if image:
                    yield image


# Example usage:
if __name__ == "__main__":
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    path_to_images: str = os.path.join(dir_path, "Dataset-Mini")
    image_loader = ImageLoader(Path(path_to_images))

    # Iterate through the images using the generator
    img: Image.Image
    for img in image_loader:
        img.show()  # Display the image (you can replace this with your own processing logic)

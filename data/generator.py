from PIL import Image

from pathlib import Path
import os


class ImageLoader:
    def __init__(self, path: os.PathLike):
        self.path: os.PathLike = path

    @staticmethod
    def load_image(image_path: Path) -> Image:
        # Load an image using PIL
        try:
            image: Image = Image.open(image_path)
            return image
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None

    def __iter__(self) -> Image:
        # Generator that yields one image at a time
        root: os.PathLike
        dirs: list[os.PathLike]
        files: list[os.PathLike]
        for root, dirs, files in os.walk(self.path):
            file: os.PathLike
            for file in files:
                file_path: str = os.path.join(root, file)
                image: Image = self.load_image(Path(file_path))
                if image:
                    yield image


# Example usage:
if __name__ == "__main__":
    dir_path: str = os.path.dirname(os.path.realpath(__file__))
    path_to_images: str = os.path.join(dir_path, "Dataset-Mini")
    image_loader = ImageLoader(Path(path_to_images))

    # Iterate through the images using the generator
    img: Image
    for img in image_loader:
        img.show()  # Display the image (you can replace this with your own processing logic)

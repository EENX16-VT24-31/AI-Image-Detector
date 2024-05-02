import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from src.CNN.model import BinaryResNet50PreTrained
from src.CNN.config import MODEL_PATH

fileToTest: str = \
    r"C:\Users\erwinia\PycharmProjects\redditScrape\reddit-wallpapers\reddit\dalle2\13bj1cx 03 Mario eating pasta.jpg"


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


if __name__ == "__main__":
    # Initialize pretrained model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: BinaryResNet50PreTrained = BinaryResNet50PreTrained().to(device)
    model.load(MODEL_PATH)
    model.eval()

    # Load image
    image: Image.Image | None = None
    try:
        image = pil_loader(fileToTest)
    except FileNotFoundError:
        print("The file wasn't found")
        exit()

    # Get prediction from network
    transform: transforms.Compose = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    image_tensor: torch.Tensor = transform(image)
    prediction: torch.Tensor = model(image_tensor[None, :].to(device).float())

    # Plot image and prediction in same figure
    plt.imshow(image)
    plt.title(f"Network output: {prediction.item():.3f}")
    plt.show()

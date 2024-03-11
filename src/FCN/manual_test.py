import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from src.FCN.model import FCN_resnet50

fileToTest: str = "C://Users//erwinia//Pictures//MjHY1ZFOGVBPubMgSbq0oby7oq93sm129HL8xUOrANU (1).jpg"


def pil_loader(path: str) -> Image.Image:
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


if __name__ == "__main__":
    # Initialize pretrained model
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()

    # Load image
    image: Image = None
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
    fig: plt.Figure
    im: plt.Subplot
    pred: plt.Subplot
    fig, (im, pred) = plt.subplots(1, 2)
    im.imshow(image)

    pred.imshow(torch.round(prediction[0][0]).to("cpu").detach().numpy())
    plt.show()

import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from src.FCN.model import FCN_resnet50
from src.FCN.calibration import platt_scale, get_platt_params

fileToTest: str = r"C:\Users\erwinia\PycharmProjects\redditScrape" \
                  r"\reddit-wallpapers\reddit\dalle2\1aoa7mi 19 Which man cave you choosing_ 🚪.jpg"


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

    platt_params = get_platt_params()
    pred_img = platt_scale(prediction[0][0], platt_params).to("cpu").detach().numpy()
    im_pred = pred.imshow(pred_img, vmin=0, vmax=1)

    plt.subplots_adjust(wspace=0.3)
    fig.colorbar(im_pred, ax=pred, shrink=0.5)
    plt.show()

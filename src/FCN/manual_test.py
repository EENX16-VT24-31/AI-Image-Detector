import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms

from src.FCN.model import FCN_resnet50
from src.FCN.calibration import platt_scale, get_platt_params

real_bird: str = r"C:\GenImage\imagenet_midjourney\val\nature\ILSVRC2012_val_00000747.JPEG"
ai_bird: str = r"C:\GenImage\imagenet_midjourney\val\ai\10_midjourney_197.png"
ai_dog: str = r"C:\GenImage\imagenet_ai_0419_sdv4\val\ai\185_sdv4_00039.png"
real_dog: str = r"C:\GenImage\imagenet_ai_0419_biggan\val\nature\ILSVRC2012_val_00000612.JPEG"
fileToTest: str = r"I:\inpainting\images\inverted-371--n02486261_8249Inpainted.png"


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
    fig: plt.Figure
    im: plt.Subplot
    pred: plt.Subplot
    # Plot image and prediction in same figure with adjusted layout
    plt.rcParams.update({"font.size": 11})
    fig, (im, pred) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [1, 1.25]})

    real_im = im.imshow(image)
    im.set_title("AI Cat with Real background")

    platt_params = get_platt_params()
    pred_img = platt_scale(prediction[0][0], platt_params).to("cpu").detach().numpy()
    im_pred = pred.imshow(pred_img, vmin=0, vmax=1)

    pred.set_title("FCN-ALL classification")

    labels = ["AI", "Real"]
    cbar = fig.colorbar(im_pred, ax=pred, shrink=0.5, location="right")
    cbar.set_ticks([0, 1])
    cbar.ax.set_yticklabels(["AI", "Real"])
    plt.show()

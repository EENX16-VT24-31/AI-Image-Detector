import torch
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import transforms
from tqdm import tqdm
from src.VIT.visiontransformer import VIT_b16
from src.data import gen_image, reddit, inpainting_loader
from src.VIT.config import GENERATORS, BASE_PATH, AI_IMAGE_PATH, NATURE_IMAGE_PATH, IMAGE_COUNT, REDDIT_PATH, \
    INPAINTING_PATH
from src.VIT.utils import set_seeds, heatmap_b16
from torcheval.metrics import BinaryConfusionMatrix
from src.VIT.calibration import get_platt_params, platt_scale

RESIZE_IMAGE_TEST = False
USE_REDDIT = False
USE_INPAINTING = False
assert [RESIZE_IMAGE_TEST, USE_REDDIT, USE_INPAINTING].count(True) <= 1

if __name__ == "__main__":

    set_seeds()
    base_path: str= BASE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VIT_b16(pretrained=True).to(device)
    datasets: gen_image.Datasets | reddit.Datasets
    inpainting_dataset: inpainting_loader.InpaintingDataset
    if USE_REDDIT:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(224, 224)
        ])
        datasets = reddit.Datasets(REDDIT_PATH, transform=transform)
    elif USE_INPAINTING:
        inpainting_dataset = inpainting_loader.InpaintingDataset(INPAINTING_PATH, "test")
    elif RESIZE_IMAGE_TEST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.Resize((224, 224))
        ])
        datasets = \
            gen_image.Datasets(BASE_PATH, generators=GENERATORS, transform=transform)
    else:
        datasets = gen_image.Datasets(BASE_PATH, generators=GENERATORS, image_count=IMAGE_COUNT)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    loss_fn = torch.nn.CrossEntropyLoss()

     # Set the model to evaluation mode
    model.eval()
    platt_params: torch.Tensor
    if not any([RESIZE_IMAGE_TEST, USE_REDDIT, USE_INPAINTING]):
        platt_params = get_platt_params(model, datasets.validation)
    else:
        platt_params = get_platt_params()

    test_set: DataLoader | Subset | inpainting_loader.InpaintingDataset
    if not USE_INPAINTING:
        test_set = datasets.testing
        inpainting_transform = None
    else:
        test_set = inpainting_dataset
        inpainting_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(224, 224)
        ])

    # Load the testing data
    # Assuming test_loader provides batches of (inputs, targets)
    test_loss: float= 0
    test_acc: float= 0

    metric: BinaryConfusionMatrix = BinaryConfusionMatrix()

    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(tqdm(test_set, "Testing model")):
            if USE_REDDIT:  # Convert label to tensor and image to 4D if using reddit dataset
                targets = torch.tensor(targets).view(1)
                inputs = inputs[None, :]
            elif USE_INPAINTING:  # Label image as AI generated and convert image to 4D tensor if using inpainting
                assert inpainting_transform
                targets = torch.tensor(0).view(1)
                inputs = inpainting_transform(inputs)[None, :]

            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            # targets = targets.view(-1, 1).expand(pred.size()).float()
            predicted_targets: torch.Tensor = torch.round(platt_scale(pred, platt_params))
            # Update BinaryConfusionMatrix
            metric.update(
                torch.LongTensor(predicted_targets.to("cpu").long()).flatten(),
                torch.LongTensor(targets.to("cpu").long()).flatten()
            )

            loss = loss_fn(pred, targets)
            test_loss += loss.item()


    m = metric.compute()
    print(m)
    accuracy: torch.Tensor = m.trace() / m.sum()
    recall: torch.Tensor = m[0, 0].item() / m[0, :].sum()
    precision: torch.Tensor = m[0, 0].item() / m[:, 0].sum()
    F1: torch.Tensor = 2 * precision * recall / (precision + recall)

    print(f"Accuracy: {accuracy.item() * 100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss / len(test_set))

    # pred_and_plot_image(model=model, class_names=classes,
    #                     image_path=AI_IMAGE_PATH,
    #                     image_size=(224, 224),
    #                     device=device
    #                     )

    # pred_and_plot_image(model=model, class_names=classes,
    #                     image_path=NATURE_IMAGE_PATH,
    #                     image_size=(224, 224),
    #                     device=device
    #                     )

    #heatmap_b16(image_path=AI_IMAGE_PATH, model=model, device=device)
    #heatmap_b16(image_path=NATURE_IMAGE_PATH, model=model, device=device)

import torch
from torcheval.metrics import BinaryConfusionMatrix
from torchvision import transforms
from tqdm import tqdm

from src.FCN.config import DATA_PATH, MAX_IMAGE_SIZE, GENERATORS, REDDIT_PATH
from src.FCN.model import FCN_resnet50
from src.FCN.calibration import platt_scale, get_platt_params
from src.data import gen_image, reddit

FULL_IMAGE_TEST: bool = True
USE_REDDIT: bool = False
assert [FULL_IMAGE_TEST, USE_REDDIT].count(True) <= 1

if __name__ == "__main__":
    # Enable freeze support for multithreading on Windows, has no effect in other operating systems
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from pth file
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()

    # Load dataset
    datasets: gen_image.Datasets | reddit.Datasets
    if FULL_IMAGE_TEST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        datasets = \
            gen_image.Datasets(DATA_PATH, generators=GENERATORS, transform=transform, batch_size=1)
    elif USE_REDDIT:
        datasets = reddit.Datasets(REDDIT_PATH)
    else:
        datasets = gen_image.Datasets(DATA_PATH, generators=GENERATORS)

    # Get platt scaling values
    platt_params = get_platt_params(model, datasets.validation)

    # Setup metrics
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)
    test_loss: float = 0.0
    metric: BinaryConfusionMatrix = BinaryConfusionMatrix()
    full_image_metric: BinaryConfusionMatrix = BinaryConfusionMatrix()

    inputs: torch.Tensor
    labels: torch.Tensor
    skipped: int = 0
    for inputs, labels in tqdm(datasets.testing, "Calculating accuracy"):
        # Due to CUDA memory constraints, some images crash the test PCs, if you have more VRAM, you can increase
        # MAX_IMAGE_SIZE
        if inputs.size()[-1] * inputs.size()[-2] > MAX_IMAGE_SIZE:
            skipped += 1
            continue

        # Get testdata
        with torch.no_grad():
            if USE_REDDIT:
                labels = torch.tensor(labels)
                inputs = inputs[None]
            inputs, labels = inputs.to(device), labels.to(device)
            outputs: torch.Tensor = model(inputs)

            full_image_labels: torch.Tensor = labels
            full_image_prediction = torch.tensor([torch.round(torch.mean(output)) for output in outputs])

            labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
            predicted_labels: torch.Tensor = torch.round(platt_scale(outputs, platt_params))

            full_image_metric.update(
                torch.LongTensor(full_image_prediction.to("cpu").long()).flatten(),
                torch.LongTensor(full_image_labels.to("cpu").long()).flatten()
            )

            # Update BinaryConfusionMatrix
            metric.update(
                torch.LongTensor(predicted_labels.to("cpu").long()).flatten(),
                torch.LongTensor(labels.to("cpu").long()).flatten()
            )

            # Update loss
            loss: torch.Tensor = loss_fn(outputs, labels)
            test_loss += loss.item()

    print(f"Skipped {skipped} images due to memory constraints")

    # Print test data
    m = metric.compute()
    print("Confusion matrix", m)
    accuracy: torch.Tensor = m.trace() / m.sum()
    recall: torch.Tensor = m[0, 0].item() / m[0, :].sum()
    precision: torch.Tensor = m[0, 0].item() / m[:, 0].sum()
    F1: torch.Tensor = 2 * precision * recall / (precision + recall)

    print(f"Accuracy: {accuracy.item() * 100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss / len(datasets.testing))

    m = full_image_metric.compute()
    accuracy = m.trace() / m.sum()
    recall = m[0, 0].item() / m[0, :].sum()
    precision = m[0, 0].item() / m[:, 0].sum()
    F1 = 2 * precision * recall / (precision + recall)

    print(f"Full Image Accuracy: {accuracy.item()*100}%")
    print(f"Full Image F1-Score: {F1.item()}")
    print(test_loss/len(datasets.testing))



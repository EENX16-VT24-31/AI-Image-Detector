import torch
from torcheval.metrics import BinaryConfusionMatrix
from torchvision import transforms
from tqdm import tqdm

from src.FCN.config import INPAINTING_PATH, MAX_IMAGE_SIZE
from src.FCN.model import FCN_resnet50
from src.FCN.calibration import platt_scale, get_platt_params
from src.data.inpainting_loader import InpaintingDataset


if __name__ == "__main__":
    # Enable freeze support for multithreading on Windows, has no effect in other operating systems
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from pth file
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()

    # Load dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    val_set = InpaintingDataset(INPAINTING_PATH, "validation", transform)
    test_set = InpaintingDataset(INPAINTING_PATH, "testing", transform)

    # Get platt scaling values
    platt_params = get_platt_params(model, val_set)

    # Setup metrics
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)
    test_loss: float = 0.0
    metric: BinaryConfusionMatrix = BinaryConfusionMatrix()
    image_metric: BinaryConfusionMatrix = BinaryConfusionMatrix()

    inputs: torch.Tensor
    labels: torch.Tensor
    skipped: int = 0
    for inputs, labels in tqdm(test_set, "Calculating accuracy"):
        # Due to CUDA memory constraints, some images crash the test PCs, if you have more VRAM, you can increase
        # MAX_IMAGE_SIZE
        if inputs.size()[-1] * inputs.size()[-2] > MAX_IMAGE_SIZE:
            skipped += 1
            continue

        # Get testdatap
        with torch.no_grad():
            inputs, labels = inputs.to(device), labels.to(device)
            outputs: torch.Tensor = model(inputs)

            predicted_labels: torch.Tensor = torch.round(platt_scale(outputs, platt_params))
            predicted_class: torch.Tensor = torch.round(platt_scale(torch.mean(outputs), platt_params))

            # Update BinaryConfusionMatrix
            metric.update(
                torch.LongTensor(predicted_labels.to("cpu").long()).flatten(),
                torch.LongTensor(labels.to("cpu").long()).flatten()
            )
            image_metric.update(
                torch.LongTensor([predicted_class.to("cpu").long()]),
                torch.LongTensor([0])  # Define correct label as AI
            )

            # Update loss
            loss: torch.Tensor = loss_fn(outputs, labels)
            test_loss += loss.item()

    print(f"Skipped {skipped} images due to memory constraints")

    # Print test data
    print("Pixel data:")
    m = metric.compute()

    accuracy: torch.Tensor = m.trace() / m.sum()
    recall: torch.Tensor = m[0, 0].item() / m[0, :].sum()
    precision: torch.Tensor = m[0, 0].item() / m[:, 0].sum()
    F1: torch.Tensor = 2 * precision * recall / (precision + recall)

    print(m)
    print(f"Accuracy: {accuracy.item() * 100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss / len(test_set))

    print("Full image data:")
    m = image_metric.compute()
    accuracy = m.trace() / m.sum()
    recall = m[0, 0].item() / m[0, :].sum()
    precision = m[0, 0].item() / m[:, 0].sum()
    F1 = 2 * precision * recall / (precision + recall)

    print(m)
    print(f"Accuracy: {accuracy.item() * 100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss / len(test_set))




import torch
from torcheval.metrics import BinaryConfusionMatrix
from torchvision import transforms
from tqdm import tqdm

from src.FCN.config import DATA_PATH, MAX_IMAGE_SIZE
from src.FCN.model import FCN_resnet50
from src.data import gen_image

FULL_IMAGE_TEST = True

if __name__ == "__main__":
    # Enable freeze support for multithreading on Windows, has no effect in other operating systems
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model from pth file
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    model.eval()

    # Load dataset
    datasets: gen_image.Datasets
    if FULL_IMAGE_TEST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        datasets = \
            gen_image.Datasets(DATA_PATH, generators=[gen_image.Generator.SD1_4], transform=transform, batch_size=1)
    else:
        datasets = gen_image.Datasets(DATA_PATH, generators=[gen_image.Generator.SD1_4])

    # Setup metrics
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)
    test_loss: float = 0.0
    metric = BinaryConfusionMatrix()

    inputs: torch.Tensor
    labels: torch.Tensor
    skipped: int = 0
    for inputs, labels in tqdm(datasets.testing):
        # Due to CUDA memory constraints, some images crash the test PCs, if you have more VRAM, you can increase
        # MAX_IMAGE_SIZE
        if inputs.size()[-1] * inputs.size()[-2] > MAX_IMAGE_SIZE:
            skipped += 1
            continue

        # Get testdata
        inputs, labels = inputs.to(device), labels.to(device)
        outputs: torch.Tensor = model(inputs)

        labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
        predicted_labels = torch.round(outputs)

        # Update BinaryConfusionMatrix
        metric.update(
            torch.LongTensor(predicted_labels.to("cpu").long()).flatten(),
            torch.LongTensor(labels.to("cpu").long()).flatten()
        )

        # Update loss
        loss: torch.Tensor = loss_fn(outputs, labels)
        loss.backward()  # Does nothing, but performance seems to be better with this line
        test_loss += loss.item()

    print(f"Skipped {skipped} images due to memory constraints")

    # Print test data
    m = metric.compute()
    accuracy = m.trace() / m.sum()
    recall = m[0, 0].item() / m[0, :].sum()
    precision = m[0, 0].item() / m[:, 0].sum()
    F1 = 2 * precision * recall / (precision + recall)

    print(f"Accuracy: {accuracy.item()*100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss/len(datasets.training))



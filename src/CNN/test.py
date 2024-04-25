import torch
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np
from torchvision import transforms

from src.data import gen_image, reddit, inpainting_loader
from src.CNN.model import BinaryResNet50PreTrained
from src.CNN.config import MODEL_PATH, DATA_PATH, GENERATORS, REDDIT_PATH, INPAINTING_PATH
from src.CNN.calibration import get_platt_params, platt_scale

FULL_IMAGE_TEST = False
USE_REDDIT = False
USE_INPAINTING = False
assert [FULL_IMAGE_TEST, USE_REDDIT, USE_INPAINTING].count(True) <= 1

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    dataset: gen_image.Datasets | reddit.Datasets
    inpainting_dataset: inpainting_loader.InpaintingDataset
    if USE_REDDIT:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(224, 224)
        ])
        dataset = reddit.Datasets(REDDIT_PATH, transform=transform)
    elif USE_INPAINTING:
        inpainting_dataset = inpainting_loader.InpaintingDataset(INPAINTING_PATH, "test")
    elif FULL_IMAGE_TEST:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = \
            gen_image.Datasets(DATA_PATH, generators=GENERATORS, transform=transform, batch_size=1)
    else:
        dataset = gen_image.Datasets(DATA_PATH, generators=GENERATORS)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: BinaryResNet50PreTrained = BinaryResNet50PreTrained().to(device)
    model.load(MODEL_PATH)  # Load stored weights

    # Evaluate the model on the testdata and calculate confusion matrix
    print("Evaluation on testset starts")
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []

    platt_params: torch.Tensor
    if not any([FULL_IMAGE_TEST, USE_REDDIT, USE_INPAINTING]):
        platt_params = get_platt_params(model, dataset.validation)
    else:
        platt_params = get_platt_params()

    test_set: DataLoader | Subset | inpainting_loader.InpaintingDataset
    if not USE_INPAINTING:
        test_set = dataset.testing
        inpainting_transform = None
    else:
        test_set = inpainting_dataset
        inpainting_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomCrop(224, 224)
        ])

    with torch.no_grad():
        for (images, labels) in tqdm(test_set, "Testing model"):
            try:
                if USE_REDDIT:  # Convert label to tensor and image to 4D if using reddit dataset
                    labels = torch.tensor(labels).view(1)
                    images = images[None, :]
                elif USE_INPAINTING:  # Label image as AI generated and convert image to 4D tensor if using inpainting
                    assert inpainting_transform
                    labels = torch.tensor(0).view(1)
                    images = inpainting_transform(images)[None, :]
                images, labels = images.to(device), labels.to(device)
                outputs: torch.Tensor = model(images)
                predicted: torch.Tensor = torch.round(platt_scale(outputs, platt_params)).long()
                true_labels.extend(labels.cpu().numpy())
                predicted_labels.extend(predicted.cpu().numpy())
            except Exception as e:
                print(f"An error occurred during training: {e}")
                continue

    # Print Confusion Matrix
    conf_matrix: list[list[float]] = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate Accuracy
    accuracy: float = np.trace(conf_matrix) / np.sum(conf_matrix)
    print("Accuracy:", accuracy)

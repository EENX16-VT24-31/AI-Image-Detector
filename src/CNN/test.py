import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import numpy as np

from src.data.gen_image import Datasets
from src.CNN.model import BinaryResNet50PreTrained
from src.CNN.config import MODEL_PATH, DATA_PATH, GENERATORS
from src.CNN.calibration import get_platt_params, platt_scale

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    dataset: Datasets = Datasets(DATA_PATH, generators=GENERATORS)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: BinaryResNet50PreTrained = BinaryResNet50PreTrained().to(device)
    model.load(MODEL_PATH)  # Load stored weights

    # Evaluate the model on the testdata and calculate confusion matrix
    print("Evaluation on testset starts")
    model.eval()
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    platt_params: torch.Tensor = get_platt_params(model, dataset.validation)
    with torch.no_grad():
        for (images, labels) in tqdm(dataset.testing):
            try:
                images, labels = images.to(device), labels.to(device)
                outputs: torch.Tensor = model(images)
                predicted: torch.Tensor = torch.round(platt_scale(outputs, platt_params)).long()
                true_labels.extend(labels.numpy())
                predicted_labels.extend(predicted.numpy())
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

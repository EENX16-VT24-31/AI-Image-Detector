import torch
from torcheval.metrics import BinaryConfusionMatrix
from tqdm import tqdm

from src.FCN.config import DATA_PATH
from src.FCN.model import FCN_resnet50
from src.data import gen_image

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)
    datasets: gen_image.Datasets = gen_image.Datasets(DATA_PATH, generators=[gen_image.Generator.SD1_4])
    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)

    test_loss: float = 0.0
    metric = BinaryConfusionMatrix()

    model.eval()

    inputs: torch.Tensor
    labels: torch.Tensor
    for inputs, labels in tqdm(datasets.testing):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs: torch.Tensor = model(inputs)
        labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
        predicted_labels = torch.round(outputs)

        metric.update(
            torch.LongTensor(predicted_labels.to("cpu").long()).flatten(),
            torch.LongTensor(labels.to("cpu").long()).flatten()
        )

        loss = loss_fn(outputs, labels)
        loss.backward()

        test_loss += loss.item()

    m = metric.compute()
    accuracy = m.trace() / m.sum()
    recall = m[0, 0].item() / m[0, :].sum()
    precision = m[0, 0].item() / m[:, 0].sum()
    F1 = 2 * precision * recall / (precision + recall)

    print(f"Accuracy: {accuracy.item()*100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss/len(datasets.training))



import torch.utils.data

import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.gen_image import Datasets
from src.CNN.model import BinaryResNet50PreTrained
from src.CNN.config import DATA_PATH, GENERATORS, LEARNING_RATE, EPOCHS, MODEL_PATH

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()

    dataset: Datasets = Datasets(DATA_PATH, generators=GENERATORS)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Create an instance of the model and define the loss function and optimizer.
    model = BinaryResNet50PreTrained().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)
    num_epochs = EPOCHS
    v_best_loss = float("inf")

    # Train the model
    print('Training started')
    for epoch in range(num_epochs):
        model.train()
        running_loss: float = 0.0
        images: torch.Tensor
        labels: torch.Tensor

        for i, (images, labels) in \
                tqdm(enumerate(dataset.training), f"Training, Epoch {epoch + 1}", total=len(dataset.training)):
            # forwardpass + backwardpass + optimization
            images, labels = images.to(device), labels.to(device)
            outputs: torch.Tensor = model(images)
            loss: torch.Tensor = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print average running loss for each 1000 consecutive batches (i.e. each iteration)
            running_loss += loss.item()
            if i % 1000 == 999:
                print('Epoch %d/%d, iteration %5d/%5d, training loss: %.3f' %
                      (epoch + 1, num_epochs, i + 1, len(dataset.training), running_loss / 1000))
                running_loss = 0.0

        model.eval()
        with torch.no_grad():
            v_correct: float = 0
            v_total: int = 0
            v_running_loss: float = 0
            v_images: torch.Tensor
            v_labels: torch.Tensor

            for v_images, v_labels in tqdm(dataset.validation, f"Validation, Epoch {epoch + 1}"):
                v_images, v_labels = v_images.to(device), v_labels.to(device).float()
                v_outputs: torch.Tensor = model(v_images)
                v_loss: torch.Tensor = criterion(v_outputs.squeeze(), v_labels)
                v_running_loss += v_loss.item()

                v_predicted: torch.Tensor = torch.round(v_outputs).long()
                v_total += v_labels.size(dim=0)
                v_correct += (v_predicted.view(v_labels.size()) == v_labels).sum().item()

            v_acc: float = (v_correct / v_total) * 100
            v_average_loss: float = v_running_loss / len(dataset.validation)
            print(f'Accuracy (on validation dataset): {v_acc:.2f}%')
            print(f'Average loss (on validation dataset): {v_average_loss:.3f}')

            # Track best performance, and save the model's state
            if v_average_loss < v_best_loss:
                torch.save(model.state_dict(), MODEL_PATH)
                print(f"New Accuracy ({v_average_loss}) > Old Accuracy ({v_best_loss}) ---> Model updated")
                v_best_loss = v_average_loss

    print('Training finished')

# Training script
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.FCN.model import FCN_resnet50
from src.FCN.config import LEARNING_RATE, EPOCHS, DATA_PATH
from src.data import gen_image

if __name__ == "__main__":
    # Enable freeze support for multithreading on Windows, has no effect in other operating systems
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    model: FCN_resnet50 = FCN_resnet50().to(device)

    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    datasets: gen_image.Datasets = \
        gen_image.Datasets(DATA_PATH, generators=[gen_image.Generator.SD1_4])

    train_loader: DataLoader = datasets.training
    val_loader: DataLoader = datasets.validation

    best_eval_loss: float = float("inf")

    for epoch_index in range(EPOCHS):
        model.train()
        running_loss: float = 0.0

        inputs: torch.Tensor
        labels: torch.Tensor
        outputs: torch.Tensor
        loss: torch.Tensor

        for inputs, labels in tqdm(train_loader, f"Training Network, Epoch {epoch_index+1}"):

            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss: float = running_loss / len(train_loader)
        print("Training loss", avg_train_loss, end="\n\n")

        # Evaluation
        running_loss = 0.0
        model.eval()

        for inputs, labels in tqdm(val_loader, f"Evaluating Network, Epoch {epoch_index + 1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)

            labels = labels.view(-1, 1, 1, 1).expand(outputs.size()).float()
            loss = loss_fn(outputs, labels)
            loss.backward()  # It might seem like this line does nothing, but performance is much better with it

            running_loss += loss.item()

        avg_val_loss: float = running_loss / len(val_loader)
        print("Validation loss:", avg_val_loss, end="\n\n")
        if avg_val_loss < best_eval_loss:
            best_eval_loss = avg_val_loss
            torch.save(model.state_dict(), "../../model/FCN_test_model.pth")
        else:
            torch.save(model.state_dict(), "../../model/FCN_test_model_overfit.pth")

    print("finished training")

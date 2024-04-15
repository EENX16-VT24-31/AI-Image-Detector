# Training script
import torch
import torchvision.transforms
from tqdm import tqdm

from src.FCN.model import FCN_resnet50
from src.FCN.config import INPAINT_LEARNING_RATE, INPAINT_EPOCHS, INPAINTING_PATH, MODEL_PATH_FINETUNED
from src.data.inpainting_loader import InpaintingDataset

if __name__ == "__main__":
    # Enable freeze support for multithreading on Windows, has no effect in other operating systems
    from multiprocessing import freeze_support
    freeze_support()

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model stored at MODEL_PATH
    model: FCN_resnet50 = FCN_resnet50(pretrained=True).to(device)

    loss_fn: torch.nn.MSELoss = torch.nn.MSELoss().to(device)
    optimizer: torch.optim.Adam = torch.optim.Adam(model.parameters(), lr=INPAINT_LEARNING_RATE)

    transform: torchvision.transforms.Compose = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 0.5)
    ])

    train_set: InpaintingDataset = InpaintingDataset(INPAINTING_PATH, "training", transform=transform)
    val_set: InpaintingDataset = InpaintingDataset(INPAINTING_PATH, "validation", transform=transform)

    best_eval_loss: float = float("inf")

    for epoch_index in range(INPAINT_EPOCHS):
        model.train()
        running_loss: float = 0.0

        input: torch.Tensor
        label: torch.Tensor
        output: torch.Tensor
        loss: torch.Tensor

        for input, label in tqdm(train_set, f"Training Network, Epoch {epoch_index+1}"):
            # Expand to 4d and move to device
            input, label = input.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(input)

            loss = loss_fn(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        avg_train_loss: float = running_loss / len(train_set)
        print("Training loss", avg_train_loss, end="\n\n")

        # Evaluation
        running_loss = 0.0
        model.eval()

        with torch.no_grad():
            for input, label in tqdm(val_set, f"Evaluating Network, Epoch {epoch_index + 1}"):
                input, label = input.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(input)


                loss = loss_fn(output, label)
                running_loss += loss.item()

        avg_val_loss: float = running_loss / len(val_set)
        print("Validation loss:", avg_val_loss, end="\n\n")
        if avg_val_loss < best_eval_loss:
            best_eval_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_PATH_FINETUNED)

    print("finished training")

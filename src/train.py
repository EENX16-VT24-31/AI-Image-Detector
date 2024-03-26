# Training script

import torch
import torchvision  # type: ignore
from torch.utils.data import DataLoader
from torchvision.transforms import v2  # type: ignore
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS

from src.model import BinaryResNet50NotPreTrained
# from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BinaryResNet50NotPreTrained().to(device)

loss_fn = torch.nn.BCELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

transforms = v2.Compose(
    [
        v2.RandomResizedCrop(size=(224, 224), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToTensor(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = torchvision.datasets.FakeData(size=1000, image_size=(3, 500, 500), num_classes=2, transform=transforms)
val_dataset = torchvision.datasets.FakeData(size=200, image_size=(3, 500, 500), num_classes=2, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# tb_writer = SummaryWriter()

best_eval_loss = float("inf")
for epoch_index in range(EPOCHS):
    print(f"Epoch {epoch_index + 1}")
    model.train(True)
    running_loss = 0.0
    train_loss = 0.0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device).reshape(-1, 1).float()

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            print("batch {} loss: {}".format(i + 1, running_loss / (i + 1)))
            tb_x = epoch_index * len(train_loader) + i + 1
            # tb_writer.add_scalar("Loss/train", train_loss, tb_x)
    train_loss = running_loss / float(len(train_loader.dataset))  # type: ignore[arg-type]

    # Evaluation
    model.eval()
    running_loss = 0.0
    for i, (inputs, labels) in enumerate(val_loader):
        inputs, labels = inputs.to(device), labels.to(device).reshape(-1, 1).float()
        print(inputs.size(), labels.size())
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        running_loss += loss.item()

    avg_val_loss: float = running_loss / float(len(val_loader.dataset))  # type: ignore[arg-type]
    if avg_val_loss < best_eval_loss:
        best_eval_loss = avg_val_loss
        torch.save(model.state_dict(), "./model/resnet50model.pth")
    # tb_writer.add_scalars(
    #     "Training vs. Validation Loss", {"Training": train_loss, "Validation": avg_val_loss}, epoch_index + 1
    # )
    # tb_writer.flush()

print("finished training")

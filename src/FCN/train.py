# Training script
import sys

import torch
from torch.utils.data import DataLoader, random_split
from torchvision.transforms import v2
from tqdm import tqdm


from src.FCN.model import FCN_test, FCN_resnet50
from src.config import LEARNING_RATE, BATCH_SIZE, EPOCHS
from src.data import universal_fake_detect



# from torch.utils.tensorboard import SummaryWriter
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = FCN_resnet50().to(device)

    loss_fn = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    transforms = v2.Compose(
        [
            v2.RandomResizedCrop(size=(224, 224), antialias=True),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToTensor(),
            # v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    datasets = universal_fake_detect.Datasets("C://Users//erwinia//Documents//progan_train", (0.02, 0.02, 0.96))
    train_loader = datasets.training()
    val_loader = datasets.validation()

    # tb_writer = SummaryWriter()

    best_eval_loss = float("inf")

    for epoch_index in range(10):
        model.train(True)
        running_loss = 0.0
        train_loss = 0.0


        for i, (inputs, labels) in tqdm(enumerate(train_loader)):
            labels = labels.view(-1, 1, 1, 1).expand(-1, 1, 224, 224).float()
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 9:
                print("  batch {} loss: {}".format(i + 1, running_loss/(i+1)))
                tb_x = epoch_index * len(train_loader) + i + 1
                # tb_writer.add_scalar("Loss/train", train_loss, tb_x)
        train_loss = running_loss / len(train_loader)
        print("Training loss", train_loss)

        # Evaluation
        model.eval()
        running_loss = 0.0

        for i, (inputs, labels) in tqdm(enumerate(val_loader)):
            labels = labels.view(-1, 1, 1, 1).expand(-1, 1, 224, 224).float()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            running_loss += loss.item()

        avg_val_loss = running_loss / len(val_loader)
        print("Validation loss:", avg_val_loss)
        if avg_val_loss < best_eval_loss:
            best_eval_loss = avg_val_loss
            torch.save(model.state_dict(), "../../model/FCN_test_model.pth")
        # tb_writer.add_scalars(
        #     "Training vs. Validation Loss", {"Training": train_loss, "Validation": avg_val_loss}, epoch_index + 1
        # )
        # tb_writer.flush()

    print("finished training")

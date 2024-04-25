
import torch
from tqdm import tqdm
from src.VIT.visiontransformer import VIT_b16
from src.data import gen_image
from src.VIT.config import LEARNING_RATE, EPOCHS, WEIGHT_DECAY, IMAGE_COUNT, BASE_PATH, GENERATORS, SAVE_PATH
from torch.utils.data import DataLoader
from src.VIT.utils import set_seeds


if __name__ == "__main__":

      set_seeds()
      base_path: str= BASE_PATH

      device = "cuda" if torch.cuda.is_available() else "cpu"
      model = VIT_b16().to(device)

      dataset = gen_image.Datasets(base_path, generators=GENERATORS, image_count=IMAGE_COUNT)

      optimizer = torch.optim.Adam(params=model.parameters(),
                                    lr=LEARNING_RATE,
                                    weight_decay=WEIGHT_DECAY)

      loss_fn = torch.nn.CrossEntropyLoss()

      classes: list[str] = dataset.classes
      training: DataLoader = dataset.training
      validation: DataLoader = dataset.validation


      model.to(device)
      print(f'Model: {model} is training.')

      for epoch in range(EPOCHS):
            print(f'\nEpochs [Current/Total]: [{epoch+1}/{EPOCHS}]')

            model.train()
            train_loss: float = 0
            train_acc: float= 0

            for batch, (X, y) in enumerate(tqdm(training, "Training Network")):
                  X, y = X.to(device), y.to(device)

                  y_pred = model(X)

                  loss = loss_fn(y_pred, y)
                  train_loss += loss.item()

                  optimizer.zero_grad()
                  loss.backward()
                  optimizer.step()

                  # Calculate and accumulate accuracy metric across all batches
                  y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
                  train_acc += (y_pred_class == y).sum().item()/len(y_pred)

            train_loss = train_loss/len(training)
            train_acc = train_acc/len(training)



            model.eval()
            val_loss: float= 0
            val_acc: float= 0

            with torch.inference_mode():
                  for batch, (X, y) in enumerate(tqdm(validation, "Evaluating Network")):
                        X, y = X.to(device), y.to(device)

                        y_pred_logits = model(X)
                        loss = loss_fn(y_pred_logits, y)
                        val_loss += loss.item()

                        # Calculate and accumulate accuracy metric across all batches
                        y_pred_labels = y_pred_logits.argmax(dim=1)
                        val_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

            val_loss = val_loss/len(validation)
            val_acc = val_acc/len(validation)

            print(f'Train Loss => {train_loss:.4f}, ',
                  f'Train Acc => {train_acc:.4f}, ',
                  f'Eval Loss => {val_loss:.4f}, ',
                  f'Eval Acc => {val_acc:.4f}')

            torch.save(model.state_dict(), SAVE_PATH+f'_{epoch}.pth')

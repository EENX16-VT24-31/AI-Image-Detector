# import torch.optim as optim
from typing import Tuple
from utils import save_model
import torch


def _train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(dataloader):
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

    train_loss = train_loss/len(dataloader)
    train_acc = train_acc/len(dataloader)

    return train_loss, train_acc

def _test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    
    model.eval()
    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred_logits = model(X)
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_labels = y_pred_logits.argmax(dim=1)
            test_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc
        


def train_model(model, train_loader, test_loader, optimizer, criterion, epochs : int, device):

    model.to(device)

    for epoch in range(epochs):
        train_loss, train_acc = _train_step(model=model, dataloader=train_loader, loss_fn=criterion, optimizer=optimizer, device=device)
        test_loss, test_acc = _test_step(model=model, dataloader=test_loader, loss_fn=criterion, device=device)

        print(f'[{epoch+1}/{epochs}]: Train Loss => {train_loss:.4f}, Train Acc => {train_acc:.4f}, Test Loss => {test_loss:.4f}, Test Acc => {test_acc:.4f}')
    
    # Save the trained model
    save_model(epochs, model, optimizer, criterion, 'trained_model.pth')

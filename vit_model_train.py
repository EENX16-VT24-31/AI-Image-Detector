import torch.optim as optim
from utils import save_model

def train_model(model, train_loader, criterion, epochs : int, lr):

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)


    for epoch in range(epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}')


    # Save the trained model
    save_model(epochs, model, optimizer, criterion, 'trained_model.pth')


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models.vision_transformer as vision_transformer
from datasets import train_loader



model = vision_transformer.vit_b_16()

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs : int = 10
for epoch in range(num_epochs):
    model.train()
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')

"""
Epoch [10/10], Loss: 0.6971055865287781
"""
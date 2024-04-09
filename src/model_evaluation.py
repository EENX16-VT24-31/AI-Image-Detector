import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
#from torch.utils.data import Dataset, DataLoader
#from torchvision import datasets
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np

# Definiera transformeringar för att förbereda bilderna
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


# Ladda CIFAR10-datasetet och definiera laddare för träning och test
trainset1 = torch.utils.data.Dataset()
trainset = torchvision.datasets.CIFAR10(root=r'\eval_data', train=True,
                                        download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True)

testset = torchvision.datasets.CIFAR10(root=r'\eval_data', train=False,
                                       download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False)

# Definiera klasserna i CIFAR10
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Definiera CNN-modellen
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Skapa en instans av modellen och definiera förlustfunktion och optimerare
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01) # lr=0.001, momentum=0.9)
num_epochs = 5
# Träna modellen

for epoch in range(num_epochs):  # loopa över datasetet flera gånger

    running_loss = 0.0
    for i, data in enumerate(trainloader):
        # få inputs; data är en lista av [inputs, labels]
        inputs, labels = data

        # nollställa parametrarna för gradienten
        #optimizer.zero_grad()

        # framåtpassning + bakåtpassning + optimering
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # nollställa parametrarna för gradienten
        optimizer.zero_grad()

        # skriv ut statistik
        running_loss += loss.item()
        if i % 2500 == 2499:    # skriv ut var 2500 minibatches
            print('Epoch %d/%d, iteration %5d/%5d, loss: %.3f' %
                  (epoch + 1, num_epochs, i + 1, len(trainloader), running_loss / 2500))
            running_loss = 0.0

    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print(f'Accuracy (on testdata): {(correct/total)*100:.2f}%')

print('Träning klar')


# Evaluate the model on the testdata and calculate confusion matrix
model.eval()
true_labels = []
predicted_labels = []
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.numpy())
        predicted_labels.extend(predicted.numpy())

# Print Confusion Matrix
conf_matrix = confusion_matrix(true_labels, predicted_labels)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate Accuracy
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
print("Accuracy:", accuracy)


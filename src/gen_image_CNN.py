import torch
import torch.utils.data

import torch.nn as nn
#import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np

from data.gen_image import Datasets
from data.gen_image import Generator

from model import BinaryResNet50PreTrained

import multiprocessing


# Define CNN-model
# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(3, 6, 5)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.conv2 = nn.Conv2d(6, 9, 5)
#         self.conv3 = nn.Conv2d(9, 12, 4)
#         self.conv4 = nn.Conv2d(12, 15, 4)
#         self.conv5 = nn.Conv2d(15, 18, 4)
#         self.fc1 = nn.Linear(18 * 4 * 4, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 1)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool(F.relu(self.conv3(x)))
#         x = self.pool(F.relu(self.conv4(x)))
#         x = self.pool(F.relu(self.conv5(x)))
#         x = x.view(-1, 18 * 4 * 4)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = torch.sigmoid(self.fc3(x))
#         return x

if __name__ == "__main__":
    multiprocessing.freeze_support()

    # Path to base directory GenImage and Selection of which Generators the dataset will consist of
    #####################################################################
    base_path: str = "C:/Kurser/Kandidatarbete/GenImage"
    all_generators = [Generator.ALL]
    sdv1_4_generator = [Generator.SD1_4]
    dataset: Datasets = Datasets(base_path, generators=sdv1_4_generator)
    #####################################################################

    # Create an instance of the model and define the loss function and optimizer.
    model = BinaryResNet50PreTrained()
    #model = CNN()
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 5
    best_v_acc = 0.0

    # Train the model
    print('Training started')
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(dataset.training):
            # forwardpass + backwardpass + optimization
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print average running loss for each 1000 consecutive batches (i.e. each iteration)
            running_loss += loss.item()
            if i % 1000 == 999:
                print('Epoch %d/%d, iteration %5d/%5d, loss: %.3f' %
                    (epoch + 1, num_epochs, i + 1, len(dataset.training), running_loss / 1000))
                running_loss = 0.0

        model.eval()
        with torch.no_grad():
            v_correct = 0
            v_total = 0
            v_running_loss = 0
            for (v_images, v_labels) in enumerate(dataset.validation):
                v_outputs = model(v_images)
                v_loss = criterion(v_outputs.squeeze(), v_labels)
                v_running_loss += v_loss.item()

                v_predicted = torch.round(v_outputs).long()
                v_total += v_labels.size(dim=0)
                v_correct += (v_predicted.view(1,32) == v_labels).sum().item()

            v_acc = (v_correct/v_total)*100
            v_average_loss = v_running_loss/len(dataset.validation)
            print(f'Accuracy (on validation dataset): {v_acc:.2f}%')
            print(f'Average loss (on validation dataset): {v_average_loss:.3f}')
            # Track best performance, and save the model's state
            if v_acc > best_v_acc:
                best_v_acc = v_acc
                #model_path = 'model_{}_{}'.format(timestamp, epoch_number) #??
                torch.save(model.state_dict(), "ResNet50_GenImage.pth")
                print("New Accuracy > Old Accuracy ---> Model updated")


    print('Training finished')
    # model.load_state_dict(torch.load("ResNet50_GenImage.pth"))

    # Evaluate the model on the testdata and calculate confusion matrix
    print("Evaluation on testset starts")
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for (images, labels) in enumerate(dataset.testing):
            try:
                outputs = model(images)
                predicted = torch.round(outputs).long()
                true_labels.extend(labels.numpy())
                predicted_labels.extend(predicted.numpy())
            except Exception as e:
                print(f"An error occurred during training: {e}")
                continue

    # Print Confusion Matrix
    conf_matrix = confusion_matrix(true_labels, predicted_labels)
    print("Confusion Matrix:")
    print(conf_matrix)

    # Calculate Accuracy
    accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
    print("Accuracy:", accuracy)



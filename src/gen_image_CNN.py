import torch
import torch.utils.data
#from torchvision import transforms
#from torchvision import datasets

import torch.nn as nn
import torch.nn.functional as F
#from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import numpy as np

#from data.universal_fake_detect import Datasets
from data.gen_image import Datasets
from data.gen_image import Generator

#from model import BinaryResNet50NotPreTrained
from model import BinaryResNet18PreTrained

import multiprocessing


# Define CNN-model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 9, 5)
        self.conv3 = nn.Conv2d(9, 12, 4)
        self.conv4 = nn.Conv2d(12, 15, 4)
        self.conv5 = nn.Conv2d(15, 18, 4)
        self.fc1 = nn.Linear(18 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.view(-1, 18 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

if __name__ == "__main__":
    multiprocessing.freeze_support()
    
    base_path: str = "C:/Kurser/Kandidatarbete/GenImage"
    dataset: Datasets = Datasets(base_path, generators=[Generator.SD1_4])

    # Skapa en instans av modellen och definiera förlustfunktion och optimerare
    #model = BinaryResNet50NotPreTrained()
    model = BinaryResNet18PreTrained()
    #model = CNN()

    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    num_epochs = 1 #5
    best_v_acc = 0

    # Train the model
    print('Training started')
    for epoch in range(num_epochs):  # loopa över datasetet flera gånger
        model.train()
        running_loss = 0.0
        j = 0
        for i, data in enumerate(dataset.training):
            try:
                model.train()
                images, labels = data
                # framåtpassning + bakåtpassning + optimering
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # skriv ut statistik
                running_loss += loss.item()
                if i % 50 == 49:    # skriv ut var 50 minibatches
                    print('Epoch %d/%d, iteration %5d/%5d, loss: %.3f' %
                        (epoch + 1, num_epochs, i + 1, len(dataset.training), running_loss / 50))
                    running_loss = 0.0
                    #j += 1
                    # if j % 5 == 4:
                    #     model.eval()
                    #     with torch.no_grad():
                    #         v_correct = 0
                    #         v_total = 0
                    #         nr_batches = 10
                    #         k = 0
                    #         v_running_loss = 0
                    #         for (v_images, v_labels) in enumerate(dataset.validation):
                    #             v_outputs = model(v_images)
                    #             v_loss = criterion(v_outputs.squeeze(), v_labels)
                    #             v_running_loss += v_loss.item()

                                #for k in range(nr_batches):
                                #(v_images, v_labels) = next(iter(dataset.validation))
                                # v_outputs = model(v_images)
                                # v_predicted = torch.round(v_outputs).long()
                                # v_total += v_labels.size(dim=0)
                                # v_correct += (v_predicted.view(1,32) == v_labels).sum().item()
                                # if k >= nr_batches:
                                #     break
                                # k += 1

                            ##v_acc = (v_correct/v_total)*100
                            ##print(f'Accuracy (on testdata): {v_acc:.2f}%')
                            # Track best performance, and save the model's state
                            # if v_acc > best_v_acc:
                            #     best_v_acc = v_acc
                            #     #model_path = 'model_{}_{}'.format(timestamp, epoch_number)
                            #     #torch.save(model.state_dict(), "ResNet50_sdv1_test.pth")
                            #     #print("New Accuracy > Old Accuracy -> Model updated")
                            #     #print("loss: %.3f" % (running_loss/len(dataset.training)))
                    #j += 1
            except Exception as e:
                print(f"An error occurred during training: {e}")
                continue


    # print('Training finished')
    # torch.save(model.state_dict(), "ResNet18_sdv1_test.pth")
    model.load_state_dict(torch.load("ResNet18_sdv1_test.pth"))

    # Evaluate the model on the testdata and calculate confusion matrix
    print("Evaluering startar")
    model.eval()
    true_labels = []
    predicted_labels = []
    with torch.no_grad():
        for i, data in enumerate(dataset.testing):
            try:
                images, labels = data
                outputs = model(images)
                predicted = torch.round(outputs).long()
                true_labels.extend(labels.numpy())
                predicted_labels.extend(predicted.numpy())
                if i % 50 == 49:    # skriv ut var 50 minibatches
                    print('Iteration %5d/%5d' %
                          (i + 1, len(dataset.testing)))
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



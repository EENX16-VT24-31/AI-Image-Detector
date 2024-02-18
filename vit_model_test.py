
import torch
from sklearn.metrics import accuracy_score

def test_model(model, test_loader, saved_model_path, criterion):
    # Load the trained parameters
    model.load_state_dict(torch.load(saved_model_path)['model_state_dict'])

    # Set the model to evaluation mode
    model.eval()

    # Load the testing data
    # Assuming test_loader provides batches of (inputs, targets)
    # Adjust this part according to your actual data loading process
    test_loss = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)  # Accumulate the test loss

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate the average test loss
    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

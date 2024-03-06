
import torch
from sklearn.metrics import accuracy_score

def validate(model, val_loader, criterion, saved_model_path=None):
    # Load the trained parameters
    if saved_model_path:
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
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)  # Accumulate the test loss

            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate the average test loss
    test_loss /= len(val_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Test Accuracy: {accuracy:.4f}")

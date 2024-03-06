# import torch.optim as optim
from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from utils import save_model
import torch
from tqdm import tqdm
from PIL import Image
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score


def _train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    
    model.train()
    train_loss, train_acc = 0, 0

    for batch, (X, y) in enumerate(tqdm(dataloader, f"Training Network")):
        X, y = X.to(device), y.to(device)

        y_pred, _ = model(X)

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
        for batch, (X, y) in enumerate(tqdm(dataloader, f"Evaluating Network")):
            X, y = X.to(device), y.to(device)

            y_pred_logits, _ = model(X)
            loss = loss_fn(y_pred_logits, y)
            test_loss += loss.item()

            # Calculate and accumulate accuracy metric across all batches
            y_pred_labels = y_pred_logits.argmax(dim=1)
            test_acc += (y_pred_labels == y).sum().item()/len(y_pred_labels)

    test_loss = test_loss/len(dataloader)
    test_acc = test_acc/len(dataloader)
    return test_loss, test_acc
        


def train_vit(model, train_loader, test_loader, optimizer, criterion, epochs : int, device):

    model.to(device)
    print(f'Model: {model} is training.')

    for epoch in range(epochs):
        print(f'\nEpochs [Current/Total]: [{epoch+1}/{epochs}]')
        train_loss, train_acc = _train_step(model=model, dataloader=train_loader, loss_fn=criterion, optimizer=optimizer, device=device)
        test_loss, test_acc = _test_step(model=model, dataloader=test_loader, loss_fn=criterion, device=device)

        print(f'Train Loss => {train_loss:.4f}, Train Acc => {train_acc:.4f}, Eval Loss => {test_loss:.4f}, Eval Acc => {test_acc:.4f}')
    
    # Save the trained model
    save_model(epochs, model, optimizer, criterion, 'trained_model.pth')


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

    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(tqdm(val_loader, f'Validating model')):
            pred, _ = model(inputs)
            loss = criterion(pred, targets)
            test_loss += loss.item() * inputs.size(0)  # Accumulate the test loss

            _, predicted = torch.max(pred, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate the average test loss
    test_loss /= len(val_loader.dataset)
    print(f"Val Loss: {test_loss:.4f}")

    # Calculate accuracy
    accuracy = accuracy_score(all_targets, all_predictions)
    print(f"Val Accuracy: {accuracy:.4f}")





def single_image_val(img_url, model, transform, classes, device):
    """
    This function is for testing purposes and proof of concept of the attention mapping
    """

    img = Image.open(img_url)
    x = transform(img)
    model.to(device)

    model.eval()
    pred_logits, attention_mat = model(x.unsqueeze(dim=0).to(device))

    print(attention_mat)
    # attention_mat = attention_matrices

    att_mat = torch.stack(attention_mat).squeeze(1)
    print(f'This is the att_mat: {att_mat}', att_mat.shape)
    # Average the attention weights across all heads.
    att_mat = torch.mean(att_mat, dim=1)
    print(f'This is the mean att_mat: {att_mat}', att_mat.shape)
    # To account for residual connections, we add an identity matrix to the
    # attention matrix and re-normalize the weights.
    residual_att = torch.eye(att_mat.size(1))
    aug_att_mat = att_mat + residual_att
    aug_att_mat = aug_att_mat / aug_att_mat.sum(dim=-1).unsqueeze(-1)
    print(aug_att_mat, aug_att_mat.shape)

    # Recursively multiply the weight matrices
    joint_attentions = torch.zeros(aug_att_mat.size())
    joint_attentions[0] = aug_att_mat[0]

    for n in range(1, aug_att_mat.size(0)):
        joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n - 1])

    # Attention from the output token to the input space.
    v = joint_attentions[-1]
    print(f'This is the v tensor: {v}', v.shape)
    grid_size = int(np.sqrt(aug_att_mat.size(-1)))
    print(grid_size)

    # Assuming mask is a NumPy array
    mask = v.reshape(grid_size, grid_size).detach().cpu().numpy()

    # Resize mask to match image size
    mask_pil = Image.fromarray(mask)
    resized_mask_pil = transforms.Resize(img.size[::-1])(mask_pil)
    resized_mask = np.array(resized_mask_pil)  # Convert back to NumPy array after resizing

    # Normalize resized mask
    resized_mask /= resized_mask.max()

    # Apply mask to the image
    img_arr = np.array(img)
    masked_img_arr = (resized_mask[..., np.newaxis] * img_arr).astype(np.uint8)
    result = Image.fromarray(masked_img_arr)

    # Visualization
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    ax1.set_title('Original')
    ax2.set_title('Attention Map')
    _ = ax1.imshow(img)
    _ = ax2.imshow(result)

    # Assuming `pred_logits` and `classes` are defined
    probs = torch.nn.Softmax(dim=-1)(pred_logits)
    top = torch.argsort(probs, dim=-1, descending=True)
    print("Prediction Label and Attention Map!\n")
    for prob, idx in zip(probs[0], top[0]):
        print(f'{prob:.5f} : {classes[idx.item()]}', end='')

    plt.show()

    # mask = v.reshape(grid_size, grid_size).detach().cpu().numpy()

    # # Resize mask to match image size
    # resized_mask = transforms.Resize(img.size[::-1])(mask)
    # resized_mask /= resized_mask.max()

    # # Apply mask to the image
    # img_arr = np.array(img)
    # masked_img_arr = (resized_mask[..., np.newaxis] * img_arr).astype(np.uint8)

    # result = Image.fromarray(masked_img_arr)

    # fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(16, 16))

    # ax1.set_title('Original')
    # ax2.set_title('Attention Map')
    # _ = ax1.imshow(img)
    # _ = ax2.imshow(result)

    # probs = torch.nn.Softmax(dim=-1)(pred_logits)
    # top = torch.argsort(probs, dim=-1, descending=True)
    # print("Prediction Label and Attention Map!\n")
    # for idx in top:
    #     print(f'{probs[0, idx.item()]:.5f} : {classes[idx.item()]}', end='')


    





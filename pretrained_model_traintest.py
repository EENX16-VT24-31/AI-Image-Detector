import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

from datasets import create_dataloaders

from vit_model_train import train_model
from vit_model_test import test_model

from utils import pred_and_plot_image




def traintest_pretrained():

    test_path : str = r'data\Dataset-Mini\test'
    train_path : str = r'data\Dataset-Mini\train'
    val_path : str = r'data\Dataset-Mini\validate'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Get pretrained weights for ViT-Base
    pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

# 2. Setup a ViT model instance with pretrained weights
    pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)

# 3. Freeze the base parameters
    for parameter in pretrained_vit.parameters():
        parameter.requires_grad = False

    # 4. Change the classifier head 
    class_names = ['0_real','1_fake']

    pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    # Print a summary using torchinfo (uncomment for actual output)
    summary(model=pretrained_vit, 
            input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
            # col_names=["input_size"], # uncomment for smaller output
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"]
    )

    pretrained_vit_transforms = pretrained_vit_weights.transforms()
    print(pretrained_vit_transforms)

    train_dataloader_pretrained, valtest_dataloader_pretrained, test_dataloader_pretrained, class_names = create_dataloaders(train_path=train_path,test_path=test_path, val_path=val_path,
                                                                                              transform=pretrained_vit_transforms,
                                                                                              batch_size=32, num_workers=0)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=pretrained_vit.parameters(), 
                                lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_model(model=pretrained_vit, train_loader=train_dataloader_pretrained, test_loader=test_dataloader_pretrained,
                optimizer=optimizer, criterion=loss_fn, epochs=10, device=device)

    # train_model(pretrained_vit, train_loader=train_dataloader_pretrained, optimizer=optimizer,
    #             criterion=loss_fn, epochs=20)
    
    # test_model(pretrained_vit, test_loader= test_dataloader_pretrained,saved_model_path='trained_model.pth', criterion=loss_fn)

    # pred_and_plot_image(pretrained_vit, class_names=class_names, image_path=r'C:\Users\maxsj\repos\AI-Image-Detector\data\Dataset-Mini\validate\0_real\00000.png', image_size=(224, 244), transform=pretrained_vit_transforms, device=device)



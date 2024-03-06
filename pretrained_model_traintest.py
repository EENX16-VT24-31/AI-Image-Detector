import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms
from torchinfo import summary

from datasets import load_images_data
from vit_helper import train_vit, single_image_val, validate
from utils import pred_and_plot_image
from myvit import VisionTransformer
from manual_transforms import create_transform



def traintest_pretrained():


    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 1. Get pretrained weights for ViT-Base
    #pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT 

    # 2. Setup a ViT model instance with pretrained weights
    # pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights).to(device)
    # vit = torchvision.models.vit_b_16().to(device)
    vit = VisionTransformer().to(device)


    # 3. Freeze the base parameters
    # for parameter in pretrained_vit.parameters():
    #     parameter.requires_grad = False

    # 4. Change the classifier head 
#     class_names = ['0_real','1_fake']

    # pretrained_vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)
    #vit.heads = nn.Linear(in_features=768, out_features=len(class_names)).to(device)

    # Print a summary using torchinfo (uncomment for actual output)
#     summary(model=vit,  
#             input_size=(32, 3, 224, 224), # (batch_size, color_channels, height, width)
#             # col_names=["input_size"], # uncomment for smaller output
#             col_names=["input_size", "output_size", "num_params", "trainable"],
#             col_width=20,
#             row_settings=["var_names"]
#     )

    # pretrained_vit_transforms = pretrained_vit_weights.transforms()
    #print(pretrained_vit_transforms)
    transform = create_transform(224)

    train_loader, test_loader, val_loader, classes = load_images_data(train_path=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\train', val_path=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val', transform=transform, size1=320, size2=320)
    # , valtest_dataloader_pretrained, , class_names= create_dataloaders(train_path=train_path,test_path=test_path, val_path=val_path, transform=pretrained_vit_transforms, batch_size=32, num_workers=0)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=vit.parameters(), #pretrained_vit.parameters(),
                                  lr=1e-3
                                  )
    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'Classes: {classes}')
    train_vit(model=vit, #pretrained_vit, 
                train_loader=train_loader, test_loader=test_loader,
                optimizer=optimizer, criterion=loss_fn, epochs=1, device=device)
    
    validate(model=vit, val_loader=val_loader, criterion=loss_fn)

    
    val_ai_path = r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\ai\VQDM_1000_200_00_017_vqdm_00020.png'
    val_nature_path = r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\nature\ILSVRC2012_val_00000744.JPEG'

    pred_and_plot_image(vit, classes, val_ai_path, (244,244), transform, device)
    pred_and_plot_image(vit, classes, val_nature_path, (244,244), transform, device)
    single_image_val(val_ai_path, vit, transform=transform, classes=['ai', 'nature'], device=device)
    single_image_val(val_nature_path, vit, transform=transform, classes=['ai', 'nature'], device=device)


    # train_model(pretrained_vit, train_loader=train_dataloader_pretrained, optimizer=optimizer,
    #             criterion=loss_fn, epochs=20)
    
    # test_model(pretrained_vit, test_loader= test_dataloader_pretrained,saved_model_path='trained_model.pth', criterion=loss_fn)

    # pred_and_plot_image(pretrained_vit, class_names=class_names, image_path=r'C:\Users\maxsj\repos\AI-Image-Detector\data\Dataset-Mini\validate\0_real\00000.png', image_size=(224, 244), transform=pretrained_vit_transforms, device=device)



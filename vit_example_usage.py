import torch
from datasets import create_loaders
from vit_helper import train, validate
from utils import pred_and_plot_image, heatmap
from visiontransformer import VisionTransformer
from manual_transforms import create_transform


# Hyperparameters
lr: float= 0.05
epochs: int= 1
tt_size: int= 100
val_size: int= 100
weight_decay: float= 0.03

def train_test(train_path: str,
                         val_path: str,
                         val_ai_path: str,
                         val_nature_path: str) -> None:

    device = "cuda" if torch.cuda.is_available() else "cpu"

    vit = VisionTransformer().to(device)

    transform = create_transform((224,224))

    train_loader, test_loader, val_loader, classes = create_loaders(train_path=train_path,
                                                                    val_path=val_path,
                                                                    transform=transform,
                                                                    size1=tt_size,
                                                                    size2=val_size)

    # Create optimizer and loss function
    optimizer = torch.optim.Adam(params=vit.parameters(),
                                 lr=lr,
                                 weight_decay=weight_decay)

    loss_fn = torch.nn.CrossEntropyLoss()
    print(f'Classes: {classes}')
    train(model=vit,
              train_loader=train_loader,
              test_loader=test_loader,
              optimizer=optimizer,
              criterion=loss_fn,
              epochs=epochs,
              device=device)

    validate(model=vit, val_loader=val_loader, criterion=loss_fn)

    pred_and_plot_image(model=vit,
                        class_names=classes,
                        image_path=val_ai_path,
                        image_size=(244,244),
                        device=device,
                        transform=transform)

    pred_and_plot_image(model=vit,
                        class_names=classes,
                        image_path=val_nature_path,
                        image_size=(244,244),
                        device=device,
                        transform=transform)

    heatmap(image_path=val_ai_path,
            model=vit,
            device=device)

    heatmap(image_path=val_nature_path,
            model=vit,
            device=device)

# Example usage of train_test_vit
if __name__ == "__main__":

    train_path: str=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\train'
    val_path: str=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val'
    val_ai_path: str= r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\ai\VQDM_1000_200_00_017_vqdm_00020.png'
    val_nature_path: str= r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\nature\ILSVRC2012_val_00000744.JPEG'

    train_test(train_path=train_path,
                   val_path=val_path,
                   val_ai_path=val_ai_path,
                   val_nature_path=val_nature_path)

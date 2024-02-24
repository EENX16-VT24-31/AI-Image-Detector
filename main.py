import torchvision.models.vision_transformer as vision_transformer
# import vit_model_train
# import vit_model_test
from datasets import create_dataloaders
# from datasets import test_loader
import torch.nn as nn
from manual_transforms import create_transform
import matplotlib.pyplot as plt
from patch_embedding import PatchEmbedding

from pretrained_model_traintest import traintest_pretrained
from utils import pred_and_plot_image

def main():

    test_path : str = r'data\Dataset-Mini\test'
    train_path : str = r'data\Dataset-Mini\train'
    val_path : str = r'data\Dataset-Mini\validate'


    BATCH_SIZE = 1
    IMG_SIZE = 224
    NUM_WORK = 0

    transform = create_transform(IMG_SIZE)

    train_loader, val_loader, test_loader, class_names = create_dataloaders(train_path=train_path,
                                                                            test_path=test_path,
                                                                            val_path=val_path,
                                                                            transform=transform,
                                                                            batch_size=BATCH_SIZE,
                                                                            num_workers=NUM_WORK
                                                                            )
    
    img_batch, label_batch = next(iter(train_loader))
    img, label = img_batch[0], label_batch[0]

    print(img.shape, label)

    plt.imshow(img.permute(1,2,0))
    plt.title(class_names[label])
    plt.axis(False)
    # plt.show()

    patcher = PatchEmbedding(in_channels=3, patch_size=16, embedding_dim=768)

    print(f'Input image shape: {img.unsqueeze(0).shape}')
    patch_embedded_img = patcher(img.unsqueeze(0))
    print(f'Output image shape: {patch_embedded_img.shape}')

    traintest_pretrained()

    # print("Detected Classes are: ", train_dataset.class_to_idx) 
    # model = vision_transformer.vit_b_16(progress=True)

    # criterion = nn.CrossEntropyLoss()

    # vit_model_train.train_model(model=model, train_loader=train_loader,criterion=criterion, epochs=10, lr=0.001)
    # vit_model_test.test_model(
    #                         model=model,
    #                         test_loader=test_loader,
    #                         saved_model_path='trained_model.pth',
    #                         criterion=criterion
    #                         )

if __name__ == "__main__":
    main()

import torchvision.models.vision_transformer as vision_transformer
import vit_model_train
import vit_model_test
from datasets import train_loader
from datasets import test_loader
import torch.nn as nn

def main():

    model = vision_transformer.vit_b_16()

    criterion = nn.CrossEntropyLoss()

    vit_model_train.train_model(model=model, train_loader=train_loader,criterion=criterion, epochs=2, lr=0.01)
    vit_model_test.test_model(
                            model=model,
                            test_loader=test_loader,
                            saved_model_path='trained_model.pth',
                            criterion=criterion
                            )

if __name__ == "__main__":
    main()

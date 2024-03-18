
import torch
from src.VIT.visiontransformer import VisionTransformer
from src.data import gen_image
from src.VIT.config import LEARNING_RATE, EPOCHS, WEIGHT_DECAY, IMAGE_COUNT, BASE_PATH, GENERATORS
from src.VIT.vit_helper import train
from torch.utils.data import DataLoader
from src.VIT.utils import set_seeds


if __name__ == "__main__":

    set_seeds()
    base_path: str= BASE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer().to(device)
    dataset = gen_image.Datasets(base_path, generators=GENERATORS, image_count=IMAGE_COUNT)

    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    loss_fn = torch.nn.CrossEntropyLoss()
    classes: list[str] = dataset.classes
    training: DataLoader = dataset.training
    validation: DataLoader = dataset.validation

    train(model=model,
          train_loader=training,
          test_loader=validation,
          optimizer=optimizer,
          criterion=loss_fn,
          epochs=EPOCHS,
          device=device)

    torch.save(model.state_dict(), "../../models/VIT_test_model.pth")

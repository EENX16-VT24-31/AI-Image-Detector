import torch
from src.VIT.visiontransformer import VisionTransformer
from src.data import gen_image
from src.VIT.config import GENERATORS, LOAD_PATH, BASE_PATH
from src.VIT.vit_helper import validate
from torch.utils.data import DataLoader
from src.VIT.utils import set_seeds

if __name__ == "__main__":

    set_seeds()
    base_path: str= BASE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VisionTransformer().to(device)
    dataset = gen_image.Datasets(base_path, generators=GENERATORS)

    model.load_state_dict(torch.load(LOAD_PATH))

    loss_fn = torch.nn.CrossEntropyLoss()
    classes: list[str] = dataset.classes
    testing: DataLoader = dataset.testing
    validate(model=model, val_loader=testing, criterion=loss_fn)



import torch
from src.VIT.visiontransformer import VIT_b16
from src.data import gen_image
from src.VIT.config import GENERATORS, BASE_PATH, AI_IMAGE_PATH, NATURE_IMAGE_PATH
from src.VIT.vit_helper import validate
from torch.utils.data import DataLoader
from src.VIT.utils import set_seeds, heatmap_b16, pred_and_plot_image

if __name__ == "__main__":

    set_seeds()
    base_path: str= BASE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VIT_b16(pretrained=True).to(device)
    dataset = gen_image.Datasets(base_path, generators=GENERATORS)

    loss_fn = torch.nn.CrossEntropyLoss()
    classes: list[str] = dataset.classes
    testing: DataLoader = dataset.testing

    validate(model=model, val_loader=testing, criterion=loss_fn, device=device)

    pred_and_plot_image(model=model, class_names=classes,
                        image_path=AI_IMAGE_PATH,
                        image_size=(224, 224),
                        device=device
                        )

    pred_and_plot_image(model=model, class_names=classes,
                        image_path=NATURE_IMAGE_PATH,
                        image_size=(224, 224),
                        device=device
                        )

    heatmap_b16(image_path=AI_IMAGE_PATH, model=model, device=device)
    heatmap_b16(image_path=NATURE_IMAGE_PATH, model=model, device=device)

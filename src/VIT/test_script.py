import torch
from tqdm import tqdm
from src.VIT.visiontransformer import VIT_b16
from src.data import gen_image
from src.VIT.config import GENERATORS, BASE_PATH, AI_IMAGE_PATH, NATURE_IMAGE_PATH, IMAGE_COUNT
from src.VIT.utils import set_seeds, heatmap_b16
from torcheval.metrics import BinaryConfusionMatrix
from src.VIT.calibration import get_platt_params, platt_scale

if __name__ == "__main__":

    set_seeds()
    base_path: str= BASE_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = VIT_b16(pretrained=True).to(device)
    datasets = gen_image.Datasets(base_path, generators=GENERATORS, image_count=IMAGE_COUNT)

    loss_fn = torch.nn.CrossEntropyLoss()
    classes: list[str] = datasets.classes

     # Set the model to evaluation mode
    model.eval()
    platt_params = get_platt_params(model, datasets.validation)

    # Load the testing data
    # Assuming test_loader provides batches of (inputs, targets)
    test_loss: float= 0
    test_acc: float= 0

    metric: BinaryConfusionMatrix = BinaryConfusionMatrix()

    with torch.inference_mode():
        for batch, (inputs, targets) in enumerate(tqdm(datasets.testing, "Testing model")):
            inputs, targets = inputs.to(device), targets.to(device)
            pred = model(inputs)
            # targets = targets.view(-1, 1).expand(pred.size()).float()
            predicted_targets: torch.Tensor = torch.round(platt_scale(pred, platt_params))
            # Update BinaryConfusionMatrix
            metric.update(
                torch.LongTensor(predicted_targets.to("cpu").long()).flatten(),
                torch.LongTensor(targets.to("cpu").long()).flatten()
            )

            loss = loss_fn(pred, targets)
            test_loss += loss.item()


    m = metric.compute()
    accuracy: torch.Tensor = m.trace() / m.sum()
    recall: torch.Tensor = m[0, 0].item() / m[0, :].sum()
    precision: torch.Tensor = m[0, 0].item() / m[:, 0].sum()
    F1: torch.Tensor = 2 * precision * recall / (precision + recall)

    print(f"Accuracy: {accuracy.item() * 100}%")
    print(f"F1-Score: {F1.item()}")
    print(test_loss / len(datasets.training))

    # pred_and_plot_image(model=model, class_names=classes,
    #                     image_path=AI_IMAGE_PATH,
    #                     image_size=(224, 224),
    #                     device=device
    #                     )

    # pred_and_plot_image(model=model, class_names=classes,
    #                     image_path=NATURE_IMAGE_PATH,
    #                     image_size=(224, 224),
    #                     device=device
    #                     )

    heatmap_b16(image_path=AI_IMAGE_PATH, model=model, device=device)
    heatmap_b16(image_path=NATURE_IMAGE_PATH, model=model, device=device)

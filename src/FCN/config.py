from src.data.gen_image import Generator

BATCH_SIZE: int = 32
EPOCHS: int = 2
LEARNING_RATE: float = 0.01
DATA_PATH: str = r"C:\GenImage"
MODEL_NAME: str = "FCN_ALL"
GENERATORS: list[Generator] = [Generator.ALL]
MODEL_PATH: str = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model" + "\\" + MODEL_NAME + ".pth"
PLATT_PATH: str = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model\platt" + "\\" + MODEL_NAME + ".pt"
MAX_IMAGE_SIZE: int = 1080 * 1920


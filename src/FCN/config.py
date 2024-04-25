from src.data.gen_image import Generator

BATCH_SIZE: int = 32

EPOCHS: int = 1
LEARNING_RATE: float = 0.001

INPAINT_EPOCHS: int = 100
INPAINT_LEARNING_RATE: float = 0.00001

DATA_PATH: str = r"C:\GenImage"
INPAINTING_PATH: str = r"I:\inpainting"
REDDIT_PATH = r"C:\Users\erwinia\PycharmProjects\redditScrape\reddit-wallpapers\reddit"
GENERATORS: list[Generator] = [Generator.ALL]

MODEL_NAME: str = "FCN_ALL_5_FINETUNED"
_BASE_PATH = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model" + "\\"
MODEL_PATH: str = _BASE_PATH + MODEL_NAME + ".pth"
MODEL_PATH_FINETUNED: str = _BASE_PATH + MODEL_NAME + "_FINETUNED.pth"
PLATT_PATH: str = _BASE_PATH + r"platt" + "\\" + MODEL_NAME + ".pt"
MAX_IMAGE_SIZE: int = 1080 * 1920


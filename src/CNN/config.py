from src.data.gen_image import Generator

EPOCHS = 2
LEARNING_RATE = 0.01

DATA_PATH = r"C:\GenImage"
REDDIT_PATH = r"C:\Users\erwinia\PycharmProjects\redditScrape\reddit-wallpapers\reddit"
INPAINTING_PATH: str = r"I:\inpainting"

MODEL_NAME = "CNN_ALL"
assert MODEL_NAME in ["CNN_SD14", "CNN_ALL"]

GENERATORS = [Generator.ALL]
MODEL_PATH = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model" + "\\" + MODEL_NAME + ".pth"
PLATT_PATH = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model\platt" + "\\" + MODEL_NAME + ".pt"

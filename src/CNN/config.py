from src.data.gen_image import Generator

EPOCHS = 10
LEARNING_RATE = 0.01

DATA_PATH = r"C:\GenImage"

MODEL_NAME = "CNN_ALL"
assert MODEL_NAME in ["CNN_SD14", "CNN_ALL"]

GENERATORS = [Generator.SD1_4] if MODEL_NAME == "CNN_SD14" else [Generator.ALL]
MODEL_PATH = r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model" + "\\" + MODEL_NAME + ".pth"

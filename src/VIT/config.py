from src.data import gen_image

LEARNING_RATE: float=0.05
EPOCHS: int=1
IMAGE_COUNT: int=100
WEIGHT_DECAY: float=0.03
LOAD_PATH: str=r"C:\Users\maxsj\models\VIT_test_model.pth"
BASE_PATH: str = r"C:\Users\maxsj\GenImage"
GENERATORS: list[gen_image.Generator]=[gen_image.Generator.VQDM]

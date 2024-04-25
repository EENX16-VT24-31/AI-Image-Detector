from src.data import gen_image

LEARNING_RATE: float=0.05
EPOCHS: int=3
IMAGE_COUNT: int | None=100
WEIGHT_DECAY: float=0.00
LOAD_PATH: str=r'C:\Users\maxsj\models\Vit_vqdm_0.pth' # Used to load the saved weights to the modelwhen testing.
SAVE_PATH: str=r'C:\Users\maxsj\models\Vit_vqdm'  # OBS! Without '.pth' Used to save each epoch during training.
PLATT_PATH: str=r'C:\Users\maxsj\models\VIT_platt_v2.pth'
BASE_PATH: str = r"C:\Users\maxsj\GenImage"
GENERATORS: list[gen_image.Generator]=[gen_image.Generator.VQDM]
AI_IMAGE_PATH: str=r'C:\Users\maxsj\GenImage\vqdm\val\ai\VQDM_1000_200_00_017_vqdm_00020.png'
NATURE_IMAGE_PATH: str=r'C:\Users\maxsj\GenImage\vqdm\val\nature\ILSVRC2012_val_00000744.JPEG'

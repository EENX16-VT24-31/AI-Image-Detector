from src.data import gen_image

LEARNING_RATE: float=0.05
EPOCHS: int=3
IMAGE_COUNT: int | None=10000
WEIGHT_DECAY: float=0.00
LOAD_PATH: str=r'C:\Users\maxsj\models\Vit_SD14_0.0001lr.pth'
BASE_PATH: str = r"C:\Users\maxsj\GenImage"
GENERATORS: list[gen_image.Generator]=[gen_image.Generator.MIDJOURNEY]
AI_IMAGE_PATH: str=r'C:\Users\maxsj\GenImage\vqdm\val\ai\VQDM_1000_200_00_017_vqdm_00020.png'
NATURE_IMAGE_PATH: str=r'C:\Users\maxsj\GenImage\vqdm\val\nature\ILSVRC2012_val_00000744.JPEG'

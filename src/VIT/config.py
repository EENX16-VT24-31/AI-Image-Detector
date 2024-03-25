from src.data import gen_image

LEARNING_RATE: float=0.0001
EPOCHS: int=10
IMAGE_COUNT: int | None = None
WEIGHT_DECAY: float=0.0001
LOAD_PATH: str=r"C:\Users\erwinia\PycharmProjects\AI-Image-Detector\src\VIT\Vit_test.pth"
BASE_PATH: str = r"F:\GenImage"
GENERATORS: list[gen_image.Generator]=[gen_image.Generator.SD1_4]
AI_IMAGE_PATH: str=r'F:\GenImage\imagenet_ai_0419_sdv4\val\ai\000_sdv4_00020.png'
NATURE_IMAGE_PATH: str=r'F:\GenImage\imagenet_ai_0419_sdv4\val\nature\ILSVRC2012_val_00000278.JPEG'


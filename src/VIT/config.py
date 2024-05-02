from src.data import gen_image

LEARNING_RATE: float=0.05
EPOCHS: int=3
IMAGE_COUNT: int | None=None
WEIGHT_DECAY: float=0.00

LOAD_PATH: str=r'C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model\ViT_ALL_10.pth'
# Used to load the saved weights to the modelwhen testing.

SAVE_PATH: str=r'C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model\Vit'
# OBS! Without '.pth' Used to save each epoch during training.

PLATT_PATH: str=r'C:\Users\erwinia\PycharmProjects\AI-Image-Detector\model\platt\ViT_ALL_10.pt'

BASE_PATH: str = r"C:\GenImage"
REDDIT_PATH: str = r"C:\Users\erwinia\PycharmProjects\redditScrape\reddit-wallpapers\reddit"
INPAINTING_PATH: str = r"I:\inpainting"

GENERATORS: list[gen_image.Generator]=[gen_image.Generator.BIGGAN]
AI_IMAGE_PATH: str=r'C:\GenImage\imagenet_ai_0419_sdv4\val\ai\185_sdv4_00039.png'
NATURE_IMAGE_PATH: str=r'C:\GenImage\imagenet_ai_0419_biggan\val\nature\ILSVRC2012_val_00000612.JPEG'

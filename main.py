from vit_example_usage import train_test
from utils import set_seeds

def main():

    set_seeds()

    train_path: str=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\train'
    val_path: str=r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val'
    val_ai_path: str= r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\ai\VQDM_1000_200_00_017_vqdm_00020.png'
    val_nature_path: str= r'C:\Users\maxsj\imagenet_ai_0419_vqdm\val\nature\ILSVRC2012_val_00000744.JPEG'

    train_test(train_path=train_path,
                   val_path=val_path,
                   val_ai_path=val_ai_path,
                   val_nature_path=val_nature_path)

if __name__ == "__main__":
    main()

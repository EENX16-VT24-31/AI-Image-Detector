import torchvision.transforms as transforms

def create_transform(img_size: int) -> transforms.Compose:

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),                # Resize the input image
        transforms.RandomHorizontalFlip(),           # Random horizontal flip
        transforms.RandomVerticalFlip(),             # Random vertical flip
        transforms.ToTensor(),                       # Convert the image to a tensor
        transforms.Normalize(                        # Normalize the pixel values
            mean=[0.485, 0.456, 0.406],               # Mean values for RGB channels
            std=[0.229, 0.224, 0.225]                 # Standard deviation values for RGB channels
        )
    ])
    return transform
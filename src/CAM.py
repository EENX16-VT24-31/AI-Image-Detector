import numpy as np
import cv2
import PIL.Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image,preprocess_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from model import BinaryResNet50NotPreTrained
import torch

print("CAM process started")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BinaryResNet50NotPreTrained().to(device)
checkpoint = torch.load('model/resnet50model.pth')
model.load_state_dict(checkpoint)
model.eval()

target = [ClassifierOutputTarget(0)]

#The layer after which we'd like to generate the activation map
target_layer = [model.resnet50.layer4]

print("model instance and target created")

# preprocess input image, get the input image tensor
img = np.array(PIL.Image.open('<path_to_image>'))
img = cv2.resize(img, (224,224))
img = np.float32(img) / 255
input_tensor = preprocess_image(img)

print("image loaded")

#generate CAM
cam = GradCAM(model=model, target_layers=target_layer)
grayscale_cams = cam(input_tensor=input_tensor, targets=target)
cam_image = show_cam_on_image(img, grayscale_cams[0, :], use_rgb=True)

cam = np.uint8(255*grayscale_cams[0, :])
cam = cv2.merge([cam, cam, cam])

print("CAM created")

#display the original image & the associated CAM
images = np.hstack((np.uint8(255*img), cam_image))
result_image = PIL.Image.fromarray(images)
result_image.show()

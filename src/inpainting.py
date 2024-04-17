from diffusers import StableDiffusionInpaintPipeline
import PIL.Image
import torch
import os

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#Initialize the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16"
    #torch_dtype=torch.float16,
).to(device)

#Directory paths
image_dir = "E:/Desktop/images and masks/real/images"
mask_dir = "E:/Desktop/images and masks/real/finalMasks"
output_dir = "E:/Desktop/images and masks/inpainting"

#Load prompts
with open("E:/Desktop/images and masks/generated_prompts.txt", "r") as file:
    prompts = file.read().splitlines()

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

#Process each image and its corresponding mask
for image_name in os.listdir(image_dir):


    if not prompts:
        print("No more prompts available.")
        break

    prompt = prompts.pop(0) 
    image_path = os.path.join(image_dir, image_name)
    mask_name = f"{os.path.splitext(image_name)[0]}-removebg-preview.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        continue

    alreadyThere = os.path.join("E:/Desktop/images and masks/inpainting",f"{image_name}Inpainted.png")
    if os.path.isfile(alreadyThere):
        print(f"File: {image_name} skipped, already processed ")
        continue

    # Load image and mask
    image = PIL.Image.open(image_path)
    mask_image = PIL.Image.open(mask_path)

    # Perform inpainting
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).images[0]

    # Save the output image
    output_path = os.path.join(output_dir, os.path.splitext(image_name)[0] + "Inpainted.png")
    output_image.save(output_path)

    print(f"Processed and saved: {output_path}")

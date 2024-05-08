from diffusers import StableDiffusionInpaintPipeline
import PIL.Image
import torch
import os

APPENDIX = "Inpainted.png"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

#Initialize the pipeline
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    revision="fp16"
    #torch_dtype=torch.float16,
).to(device)

#Directory paths
image_dir = r"C:\Users\erwinia\Downloads\InpaintingResources\images"
mask_dir = r"C:\Users\erwinia\Downloads\InpaintingResources\labelsInverted"
output_dir = r"C:\Users\erwinia\Downloads\InpaintingResources\output"

#Load prompts
with open(r"C:\Users\erwinia\Downloads\InpaintingResources\prompts\generated_prompts2.txt", "r") as file:
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
    mask_name = f"inverted-{os.path.splitext(image_name)[0]}-removebg-preview.png"
    mask_path = os.path.join(mask_dir, mask_name)

    if not os.path.exists(mask_path):
        continue

    alreadyThere = os.path.join(output_dir, f"inverted-{os.path.splitext(image_name)[0]}{APPENDIX}")
    if os.path.isfile(alreadyThere):
        print(f"File: {image_name} skipped, already processed ")
        continue

    # Load image and mask
    image = PIL.Image.open(image_path)
    mask_image = PIL.Image.open(mask_path)

    # Perform inpainting
    output_image = pipe(prompt=prompt, image=image, mask_image=mask_image).inputs[0]

    # Save the output image
    output_path = os.path.join(output_dir, f"inverted-{os.path.splitext(image_name)[0]}{APPENDIX}")
    output_image.save(output_path)

    print(f"Processed and saved: {output_path}")

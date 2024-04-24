from PIL import Image
import os


def invert_image_color(input_path, output_path):
    files = os.listdir(input_path)

    for file in files:
        file_path = os.path.join(input_path, file)

        if file.lower().endswith(('.png')):
            with Image.open(file_path) as img:
                inverted_img = Image.eval(img, lambda x: 255 - x)

                output_file_path = os.path.join(output_path, f"inverted-{file}")

                inverted_img.save(output_file_path)
                print(f'Inverted image saved to {output_file_path}')


if __name__ == "__main__":
    input_dir = r"I:\inpainting\labels"
    output_dir = r"C:\Users\erwinia\Downloads\InpaintingResources\labelsInverted"

    invert_image_color(input_dir, output_dir)
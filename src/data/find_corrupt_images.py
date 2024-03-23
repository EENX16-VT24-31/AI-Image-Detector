import os
from PIL import Image

def check_for_corrupt_files(folder_path):
    corrupt_files = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Check if the file is an image (you can adjust this condition as needed)
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            try:
                # Attempt to open the image file
                with Image.open(file_path) as img:
                    # Check if the image can be opened without errors
                    img.verify()
            except (IOError, SyntaxError, OSError, Image.DecompressionBombError, Image.UnidentifiedImageError) as e:
                # Handle the error gracefully
                print(f"Error: {e} - {file_path} is corrupt.")
                corrupt_files.append(file_path)
    
    return corrupt_files

if __name__ == "__main__":
    folder_path = "C:/Kurser/Kandidatarbete/GenImage/sdv4/train/nature"
    #folder_path = "C:/Kurser/Kandidatarbete/GenImage/sdv4/val/nature"
    corrupt_files = check_for_corrupt_files(folder_path)
    
    if corrupt_files:
        print(f"Found {len(corrupt_files)} corrupt file(s):")
        for file_path in corrupt_files:
            print(file_path)
    else:
        print("No corrupt files found.")

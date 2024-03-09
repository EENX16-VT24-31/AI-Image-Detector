import os

with open("corrupt.txt") as F:
    corrupt_files: list[str] = F.readlines()
    # Remove new_lines
    corrupt_files = [file[:-1] for file in corrupt_files][:-1]

base_path: str = "E://GenImage"
file: str
for file in corrupt_files:
    try:
        os.remove(os.path.join(base_path, file))
    except OSError:
        print(file, "Doesn't exist")

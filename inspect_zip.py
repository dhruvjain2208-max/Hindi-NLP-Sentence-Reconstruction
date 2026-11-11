import zipfile
from pathlib import PurePosixPath

ZIP_PATH = "input.zip"

try:
    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

        files = [
            info for info in zip_file.infolist()
            if not info.is_dir()
        ]

        print("========== DATASET INFORMATION ==========")

        print(f"\nTotal files: {len(files)}")

        print("\nFile details:")

        for file in files:
            filename = PurePosixPath(file.filename)

            print(f"\nName: {filename.name}")
            print(f"Format: {filename.suffix}")
            print(f"Size: {file.file_size / 1024:.2f} KB")

        print("\n========== INSPECTION COMPLETE ==========")

except FileNotFoundError:
    print("ZIP file not found. Check its location.")

except zipfile.BadZipFile:
    print("The supplied file is not a valid ZIP archive.")
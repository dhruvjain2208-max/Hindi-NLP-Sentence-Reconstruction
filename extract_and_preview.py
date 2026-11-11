import zipfile
from pathlib import Path, PurePosixPath

ZIP_PATH = Path("input.zip")
OUTPUT_DIR = Path("data/raw")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_files():

    if not ZIP_PATH.exists():
        print("ZIP file not found!")
        return

    with zipfile.ZipFile(ZIP_PATH, "r") as zip_file:

        for info in zip_file.infolist():

            if info.is_dir():
                continue

            file_path = PurePosixPath(info.filename)

            # Extract only text files
            if file_path.suffix.lower() != ".txt":
                continue

            # Avoid unsafe paths
            if file_path.is_absolute() or ".." in file_path.parts:
                print("Skipping unsafe path:", info.filename)
                continue

            destination = OUTPUT_DIR.joinpath(*file_path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)

            with zip_file.open(info) as source:
                destination.write_bytes(source.read())

            print("Extracted:", destination)

    print("\nExtraction completed!")


def inspect_files():

    print("\n========== TEXT INSPECTION ==========")

    for file_path in sorted(OUTPUT_DIR.rglob("*.txt")):

        raw_data = file_path.read_bytes()

        try:
            text = raw_data.decode("utf-8-sig")

        except UnicodeDecodeError:
            try:
                text = raw_data.decode("utf-16")
            except UnicodeDecodeError:
                print(f"\nCould not decode: {file_path}")
                continue

        print("\n" + "=" * 50)

        print("FILE:", file_path.name)
        print("Characters:", len(text))
        print("Words:", len(text.split()))
        print("Lines:", len(text.splitlines()))

        print("\nFIRST 500 CHARACTERS:\n")

        print(text[:500])

    print("\n========== INSPECTION COMPLETE ==========")


if __name__ == "__main__":

    extract_files()
    inspect_files()
from pathlib import Path
import unicodedata

DATA_DIR = Path("data/raw")


def analyze_file(file_path):

    text = file_path.read_text(encoding="utf-8")

    words = text.split()

    # Count Unicode punctuation characters
    punctuation = [
        char for char in text
        if unicodedata.category(char).startswith("P")
    ]

    # Count unique words
    unique_words = set(words)

    print("\n" + "=" * 60)

    print("FILE:", file_path.name)

    print("Characters:", len(text))
    print("Total words:", len(words))
    print("Unique words:", len(unique_words))
    print("Lines:", len(text.splitlines()))
    print("Punctuation characters:", len(punctuation))

    print("\nFirst 250 characters:")
    print(text[:250])

    print("\nLast 150 characters:")
    print(text[-150:])


def main():

    files = sorted(DATA_DIR.rglob("*.txt"))

    print("Total text files:", len(files))

    for file_path in files:
        analyze_file(file_path)


if __name__ == "__main__":
    main()
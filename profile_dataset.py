from pathlib import Path
import re

DATA_DIR = Path("data/raw")


def profile_file(file_path):

    text = file_path.read_text(encoding="utf-8")
    words = text.split()

    # Find NULL tokens
    null_positions = [
        i for i, word in enumerate(words)
        if word.upper() == "NULL"
    ]

    # Find numeric tokens
    numbers = [
        word for word in words
        if any(char.isdigit() for char in word)
    ]

    # Find English/Latin words
    english_words = re.findall(
        r"[A-Za-z]+",
        text
    )

    print("\n" + "=" * 60)

    print("FILE:", file_path.name)

    print("Total words:", len(words))
    print("NULL occurrences:", len(null_positions))
    print("Numeric tokens:", len(numbers))
    print("English/Latin words:", len(english_words))

    if null_positions:

        print("\nContexts containing NULL:")

        for position in null_positions:

            start = max(0, position - 8)
            end = min(len(words), position + 9)

            context = " ".join(words[start:end])

            print("\n", context)

    if numbers:

        print("\nNumeric tokens:")
        print(numbers[:30])

    if english_words:

        print("\nEnglish/Latin words:")
        print(english_words[:30])


def main():

    files = sorted(DATA_DIR.rglob("*.txt"))

    for file_path in files:
        profile_file(file_path)


if __name__ == "__main__":
    main()
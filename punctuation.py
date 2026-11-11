from pathlib import Path
import unicodedata
from collections import Counter

ORIGINAL = Path(
    "data/processed/normalized/"
    "Sample-Raw-No-Punctuations/sample-1.txt"
)

PUNCTUATED = Path(
    "data/processed/punctuation_preview/sample-1.txt"
)


def remove_punctuation(text):

    result = ""

    for char in text:

        if not unicodedata.category(char).startswith("P"):
            result += char

    return " ".join(result.split())


def main():

    original = ORIGINAL.read_text(encoding="utf-8")
    output = PUNCTUATED.read_text(encoding="utf-8")

    print("========== WORD PRESERVATION ==========")

    original_clean = remove_punctuation(original)
    output_clean = remove_punctuation(output)

    if original_clean == output_clean:
        print("PASS: Original words are preserved.")
    else:
        print("WARNING: Text differs after removing punctuation.")

    print("\nOriginal word count:", len(original.split()))
    print("Output word count:", len(output.split()))

    print("\n========== PUNCTUATION STATISTICS ==========")

    punctuation = Counter(
        char for char in output
        if unicodedata.category(char).startswith("P")
    )

    for mark, count in punctuation.most_common():

        print(f"{mark} : {count}")

    print("\nTotal punctuation marks:", sum(punctuation.values()))

    print("\n========== FIRST 10 PUNCTUATION CONTEXTS ==========")

    displayed = 0

    for index, char in enumerate(output):

        if unicodedata.category(char).startswith("P"):

            left = output[max(0, index - 35):index]
            right = output[index + 1:index + 36]

            print(f"\nBefore: {left}")
            print(f"Mark: {char}")
            print(f"After: {right}")

            displayed += 1

            if displayed == 10:
                break


if __name__ == "__main__":
    main()
    
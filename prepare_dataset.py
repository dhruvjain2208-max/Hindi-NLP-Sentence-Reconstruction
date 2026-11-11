from pathlib import Path
import unicodedata
import re
import csv

SOURCE_DIR = Path("data/raw")

OUTPUT_DIR = Path("data/processed/normalized")

SUMMARY_FILE = Path("data/processed/dataset_summary.csv")


def normalize_text(text):

    # Standardize Unicode representation
    text = unicodedata.normalize("NFC", text)

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def process_dataset():

    files = sorted(SOURCE_DIR.rglob("*.txt"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_FILE.parent.mkdir(parents=True, exist_ok=True)

    summary = []

    for file_path in files:

        # Read original file
        original_text = file_path.read_text(
            encoding="utf-8-sig"
        )

        # Normalize
        processed_text = normalize_text(original_text)

        # Preserve directory structure
        relative_path = file_path.relative_to(SOURCE_DIR)

        output_path = OUTPUT_DIR / relative_path

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # Save processed copy
        output_path.write_text(
            processed_text,
            encoding="utf-8"
        )

        words = processed_text.split()

        null_count = sum(
            1 for word in words
            if word.upper() == "NULL"
        )

        english_words = [
            word for word in words
            if re.search(r"[A-Za-z]", word)
            and word.upper() != "NULL"
        ]

        numeric_words = [
            word for word in words
            if any(char.isdigit() for char in word)
        ]

        summary.append({

            "filename": file_path.name,
            "characters": len(processed_text),
            "words": len(words),
            "null_count": null_count,
            "english_terms": len(english_words),
            "numeric_tokens": len(numeric_words)

        })

        print(f"Processed: {file_path.name}")

    # Generate dataset report
    with open(
        SUMMARY_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as csvfile:

        fieldnames = summary[0].keys()

        writer = csv.DictWriter(
            csvfile,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(summary)

    print("\nPreprocessing completed!")
    print("Total files processed:", len(summary))
    print("Output directory:", OUTPUT_DIR)
    print("Dataset report:", SUMMARY_FILE)


if __name__ == "__main__":

    process_dataset()
import re
import json

from pathlib import Path


# Get project root automatically
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# File paths
INPUT_FILE = (
    PROJECT_ROOT
    / "data/processed/punctuation_preview/sample-1.txt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/processed/segmented"
)


def split_sentences(text):

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # Split after terminal punctuation
    pattern = (
        r"(?<=[।॥?!])\s+"
        r"|(?<=\.)\s+(?=[A-Z\u0900-\u097F])"
    )

    parts = re.split(pattern, text)

    sentences = [
        sentence.strip()
        for sentence in parts
        if sentence.strip()
    ]

    return sentences


def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file missing: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    text = INPUT_FILE.read_text(
        encoding="utf-8"
    )

    sentences = split_sentences(text)

    records = []

    for index, sentence in enumerate(sentences, start=1):

        records.append({
            "sentence_id": index,
            "text": sentence
        })

    # Save JSON
    output_file = (
        OUTPUT_DIR / "sample-1_sentences.json"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            records,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Display results
    print("\n========== SEGMENTATION RESULTS ==========")

    print("Total sentences:", len(sentences))

    for record in records[:10]:

        print(
            f"\nSentence {record['sentence_id']}:"
        )

        print(record["text"])

    print("\nSegmentation completed!")

    print("JSON saved:", output_file)


if __name__ == "__main__":
    main()
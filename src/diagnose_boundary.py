import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = (
    ROOT / "data/processed/normalized/"
    "Sample-Raw-No-Punctuations/sample-1.txt"
)

PUNCT_FILE = (
    ROOT / "data/processed/punctuation_preview/"
    "sample-1.txt"
)

JSON_FILE = (
    ROOT / "data/processed/segmented/"
    "sample-1_sentences.json"
)


def show_context(text, position, radius=100):

    start = max(0, position - radius)
    end = min(len(text), position + radius)

    return text[start:end]


def main():

    records = json.loads(
        JSON_FILE.read_text(encoding="utf-8")
    )

    raw = RAW_FILE.read_text(encoding="utf-8")

    punctuated = PUNCT_FILE.read_text(encoding="utf-8")

    # Find sentence 43
    target = next(
        record["text"]
        for record in records
        if record["sentence_id"] == 43
    )

    print("========== TARGET SENTENCE ==========")
    print(target)

    print("\n========== PUNCTUATED TEXT CONTEXT ==========")

    search_position = 0

    while True:

        position = punctuated.find(target, search_position)

        if position == -1:
            break

        print("\nContext:")
        print(show_context(punctuated, position))

        search_position = position + len(target)

    print("\n========== RAW TEXT: GARH OCCURRENCES ==========")

    search_position = 0
    count = 0

    while True:

        position = raw.find("गढ़", search_position)

        if position == -1:
            break

        count += 1

        print(f"\nOccurrence {count}:")
        print(show_context(raw, position, 70))

        search_position = position + 3

        if count >= 20:
            print("\nShowing first 20 occurrences.")
            break


if __name__ == "__main__":
    main()

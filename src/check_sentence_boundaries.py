import json

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

JSON_FILE = (
    ROOT / "data/processed/segmented/"
    "sample-1_sentences.json"
)


def main():

    with open(
        JSON_FILE,
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    print("Total sentences:", len(records))

    # ---------------------------------
    # 1. Find suspiciously short sentences
    # ---------------------------------

    print("\n========== SHORT SENTENCES ==========")

    for record in records:

        word_count = len(record["text"].split())

        if word_count <= 2:

            print(
                f"\nSentence {record['sentence_id']}"
            )

            print("Word count:", word_count)

            print("Text:", record["text"])

            # Show surrounding sentences
            current_id = record["sentence_id"]

            print("\nSurrounding context:")

            for surrounding in records:

                sid = surrounding["sentence_id"]

                if abs(sid - current_id) <= 2:

                    print(
                        f"S{sid}: {surrounding['text']}"
                    )

    # ---------------------------------
    # 2. Locate NULL tokens
    # ---------------------------------

    print("\n========== NULL SENTENCES ==========")

    for record in records:

        if "NULL" in record["text"]:

            print(
                f"\nS{record['sentence_id']}: "
                f"{record['text']}"
            )

    print("\n========== ANALYSIS COMPLETE ==========")


if __name__ == "__main__":
    main()
import csv
import json

from pathlib import Path


# =====================================
# FILE PATHS
# =====================================

ROOT = Path(__file__).resolve().parents[1]

SENTENCE_FILE = (
    ROOT / "data/processed/segmented/"
    "sample-1_sentences.json"
)

PAIR_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "pairwise_similarity.csv"
)


# =====================================
# LOAD DATA
# =====================================

def load_sentences():

    with open(SENTENCE_FILE, encoding="utf-8") as file:
        records = json.load(file)

    return records


def load_similarity_pairs():

    pairs = []

    with open(
        PAIR_FILE,
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            pairs.append({
                "a": int(row["sentence_a_id"]),
                "b": int(row["sentence_b_id"]),
                "score": float(row["similarity"])
            })

    return pairs


# =====================================
# ANALYSE SENTENCE LENGTH
# =====================================

def analyse_lengths(records):

    print("\n========== SENTENCE LENGTH ANALYSIS ==========")

    for record in records:

        words = record["text"].split()

        word_count = len(words)

        if word_count > 30 or word_count < 5:

            print(f"\nSentence ID: {record['sentence_id']}")
            print("Word count:", word_count)
            print("Text:", record["text"])


# =====================================
# FIND STRONGEST PAIRS
# =====================================

def analyse_strongest_pairs(records, pairs):

    sentence_map = {
        record["sentence_id"]: record["text"]
        for record in records
    }

    print("\n========== TOP 15 SENTENCE PAIRS ==========")

    pairs.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    for pair in pairs[:15]:

        a = pair["a"]
        b = pair["b"]

        print("\n-----------------------------------")

        print(f"Sentence {a}:")
        print(sentence_map[a])

        print(f"\nSentence {b}:")
        print(sentence_map[b])

        print("\nSimilarity:", round(pair["score"], 4))


# =====================================
# MAIN
# =====================================

def main():

    records = load_sentences()

    pairs = load_similarity_pairs()

    print("Total sentences:", len(records))
    print("Total unique pairs:", len(pairs))

    analyse_lengths(records)

    analyse_strongest_pairs(
        records,
        pairs
    )


if __name__ == "__main__":
    main()
    
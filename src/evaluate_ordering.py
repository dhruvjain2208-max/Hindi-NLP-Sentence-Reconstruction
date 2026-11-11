import json
import csv

from pathlib import Path


# =====================================
# 1. FILE PATHS
# =====================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "beam_search_results/beam_search_candidates.json"
)

SCORE_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "ranked_transitions.csv"
)

OUTPUT_DIR = (
    ROOT / "data/processed/evaluation"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "manual_order_review.csv"
)


# =====================================
# 2. LOAD BEAM SEARCH RESULTS
# =====================================

def load_candidates():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Run beam_search_ordering.py first."
        )

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as file:

        candidates = json.load(file)

    if not candidates:
        raise ValueError("No candidate sequences found.")

    return candidates


# =====================================
# 3. LOAD TRANSITION SCORES
# =====================================

def load_scores():

    scores = {}

    with open(
        SCORE_FILE,
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            key = (
                int(row["sentence_a_id"]),
                int(row["sentence_b_id"])
            )

            scores[key] = float(
                row["transition_score"]
            )

    return scores


# =====================================
# 4. VERIFY THE BEST SEQUENCE
# =====================================

def validate_sequence(candidate):

    ids = candidate["sentence_ids"]

    print("\n========== SEQUENCE VALIDATION ==========")

    print("Total sentences:", len(ids))

    print("Unique sentence IDs:", len(set(ids)))

    print("Repeated sentences:", len(ids) != len(set(ids)))

    print(
        "Sequence complete:",
        len(ids) == 49 and len(set(ids)) == 49
    )

    if len(ids) != 49 or len(set(ids)) != 49:
        raise ValueError(
            "The sequence is incomplete or contains duplicates."
        )


# =====================================
# 5. GENERATE MANUAL REVIEW CSV
# =====================================

def generate_review(candidate, scores):

    ids = candidate["sentence_ids"]

    sentences = candidate["sentences"]

    rows = []

    for index in range(len(ids) - 1):

        source_id = ids[index]

        target_id = ids[index + 1]

        transition = (
            source_id,
            target_id
        )

        score = scores.get(transition)

        rows.append({

            "position": index + 1,

            "sentence_a_id": source_id,

            "sentence_a": sentences[index],

            "sentence_b_id": target_id,

            "sentence_b": sentences[index + 1],

            "transition_score": score,

            "human_label": "",

            "comments": ""

        })

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        OUTPUT_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys()
        )

        writer.writeheader()

        writer.writerows(rows)

    print("\nManual review generated!")

    print("Total transitions:", len(rows))

    print("Output:", OUTPUT_FILE)


# =====================================
# 6. MAIN
# =====================================

def main():

    candidates = load_candidates()

    scores = load_scores()

    best = candidates[0]

    validate_sequence(best)

    generate_review(best, scores)

    print("\n========== BEST CANDIDATE ==========")

    print("Rank:", best["rank"])

    print(
        "Mean transition score:",
        best["mean_transition_score"]
    )


if __name__ == "__main__":
    main()
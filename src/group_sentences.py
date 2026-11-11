import json

from pathlib import Path


# =====================================
# 1. PATH CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "beam_search_results/beam_search_candidates.json"
)

OUTPUT_DIR = (
    ROOT / "data/processed/translation_input"
)

OUTPUT_JSON = (
    OUTPUT_DIR / "sample-1_groups.json"
)


# =====================================
# 2. CONFIGURATION
# =====================================

GROUP_SIZE = 5


# =====================================
# 3. LOAD BEST ORDERED SEQUENCE
# =====================================

def load_best_sequence():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Beam Search output not found. "
            "Run beam_search_ordering.py first."
        )

    with open(INPUT_FILE, encoding="utf-8") as file:

        candidates = json.load(file)

    if not candidates:
        raise ValueError("No candidate sequences found.")

    best = candidates[0]

    sentence_ids = best["sentence_ids"]
    sentences = best["sentences"]

    if len(sentence_ids) != len(sentences):
        raise ValueError(
            "Sentence IDs and sentence texts do not match."
        )

    if len(set(sentence_ids)) != len(sentence_ids):
        raise ValueError(
            "Duplicate sentence IDs detected."
        )

    return best


# =====================================
# 4. GROUP SENTENCES
# =====================================

def create_groups(best):

    sentence_ids = best["sentence_ids"]
    sentences = best["sentences"]

    groups = []

    for start in range(0, len(sentences), GROUP_SIZE):

        end = start + GROUP_SIZE

        group_sentences = []

        for index in range(start, min(end, len(sentences))):

            group_sentences.append({

                "position": index + 1,

                "sentence_id": sentence_ids[index],

                "text": sentences[index]

            })

        groups.append({

            "group_id": len(groups) + 1,

            "sentence_count": len(group_sentences),

            "sentences": group_sentences

        })

    return groups


# =====================================
# 5. SAVE GROUPS
# =====================================

def save_groups(groups, best):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    result = {

        "document": "sample-1",

        "ordering_method": "MiniLM + heuristic scoring + Beam Search",

        "ordering_status": "UNVERIFIED_CANDIDATE",

        "candidate_rank": best["rank"],

        "mean_transition_score": best["mean_transition_score"],

        "total_sentences": sum(
            group["sentence_count"]
            for group in groups
        ),

        "groups": groups

    }

    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Also create individual text files
    for group in groups:

        group_id = group["group_id"]

        output_file = (
            OUTPUT_DIR /
            f"sample-1_group-{group_id:02d}.txt"
        )

        text = " ".join(
            sentence["text"]
            for sentence in group["sentences"]
        )

        output_file.write_text(
            text,
            encoding="utf-8"
        )

    return result


# =====================================
# 6. MAIN
# =====================================

def main():

    print("Loading ordered sentences...")

    best = load_best_sequence()

    print("Best candidate rank:", best["rank"])

    print(
        "Mean transition score:",
        best["mean_transition_score"]
    )

    groups = create_groups(best)

    result = save_groups(groups, best)

    print("\n========== GROUPING RESULTS ==========")

    print("Total sentences:", result["total_sentences"])

    print("Total groups:", len(groups))

    for group in groups:

        print(
            f"\nGroup {group['group_id']}: "
            f"{group['sentence_count']} sentences"
        )

        for sentence in group["sentences"]:

            print(
                f"S{sentence['sentence_id']}: "
                f"{sentence['text']}"
            )

    print("\nGrouping completed!")

    print("JSON:", OUTPUT_JSON)

    print("Text files:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
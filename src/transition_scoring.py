import csv

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "directed_pair_features.csv"
)

OUTPUT_DIR = (
    ROOT / "data/processed/ordering_analysis"
)

ALL_SCORES_FILE = (
    OUTPUT_DIR / "ranked_transitions.csv"
)

TOP_CANDIDATES_FILE = (
    OUTPUT_DIR / "top_ranked_successors.csv"
)


# ==========================================
# 2. FEATURE WEIGHTS
# ==========================================

WEIGHTS = {

    "semantic_similarity": 0.54,

    "content_jaccard": 0.18,

    "leading_overlap": 0.08,

    "reference": 0.08,

    "temporal": 0.06,

    "connective": 0.04,

    "number": 0.01,

    "latin": 0.01

}


# ==========================================
# 3. UTILITY FUNCTIONS
# ==========================================

def clamp(value, minimum=0.0, maximum=1.0):

    return max(
        minimum,
        min(maximum, value)
    )


def to_float(value):

    return float(value or 0)


# ==========================================
# 4. TRANSITION SCORING
# ==========================================

def calculate_transition_score(row):

    semantic = clamp(
        to_float(row["semantic_similarity"])
    )

    content = clamp(
        to_float(row["content_jaccard"])
    )

    leading = clamp(
        to_float(row["leading_overlap_count"]) / 3
    )

    reference = clamp(
        to_float(row["b_starts_with_reference"])
    )

    temporal = clamp(
        to_float(row["b_starts_with_temporal"])
    )

    connective = clamp(
        to_float(row["b_starts_with_connective"])
    )

    number = clamp(
        to_float(row["shared_number_count"])
    )

    latin = clamp(
        to_float(row["shared_latin_count"])
    )

    score = (

        WEIGHTS["semantic_similarity"] * semantic

        + WEIGHTS["content_jaccard"] * content

        + WEIGHTS["leading_overlap"] * leading

        + WEIGHTS["reference"] * reference

        + WEIGHTS["temporal"] * temporal

        + WEIGHTS["connective"] * connective

        + WEIGHTS["number"] * number

        + WEIGHTS["latin"] * latin

    )

    return round(score, 5)


# ==========================================
# 5. MAIN PROCESSING
# ==========================================

def main():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Feature file not found: {INPUT_FILE}\n"
            "Run directed_pair_features.py first."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    transitions = []

    # Load all directed transitions
    with open(
        INPUT_FILE,
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            score = calculate_transition_score(row)

            row["transition_score"] = score

            transitions.append(row)

    # Sort all transitions
    transitions.sort(
        key=lambda row: row["transition_score"],
        reverse=True
    )

    # ======================================
    # SAVE ALL SCORES
    # ======================================

    with open(
        ALL_SCORES_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=transitions[0].keys()
        )

        writer.writeheader()

        writer.writerows(transitions)

    # ======================================
    # TOP 5 SUCCESSORS PER SENTENCE
    # ======================================

    grouped = {}

    for row in transitions:

        sentence_id = row["sentence_a_id"]

        if sentence_id not in grouped:

            grouped[sentence_id] = []

        if len(grouped[sentence_id]) < 5:

            grouped[sentence_id].append(row)

    top_candidates = []

    for candidates in grouped.values():

        top_candidates.extend(candidates)

    with open(
        TOP_CANDIDATES_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=top_candidates[0].keys()
        )

        writer.writeheader()

        writer.writerows(top_candidates)

    # ======================================
    # DISPLAY RESULTS
    # ======================================

    print("\n========== TRANSITION SCORING ==========")

    print("Total transitions:", len(transitions))

    print("Top candidates per sentence: 5")

    print("\n========== SAMPLE RESULTS ==========")

    important_ids = [20, 21, 33]

    for sentence_id in important_ids:

        print(f"\nSENTENCE {sentence_id}")

        candidates = grouped.get(
            str(sentence_id),
            []
        )

        for candidate in candidates:

            print("\nCandidate:", candidate["sentence_b_id"])

            print(candidate["sentence_b"])

            print(
                "Similarity:",
                candidate["semantic_similarity"]
            )

            print(
                "Transition score:",
                candidate["transition_score"]
            )

    print("\n========== OUTPUT FILES ==========")

    print("All scores:", ALL_SCORES_FILE)

    print("Top candidates:", TOP_CANDIDATES_FILE)

    print("\nScoring completed!")


if __name__ == "__main__":

    main()
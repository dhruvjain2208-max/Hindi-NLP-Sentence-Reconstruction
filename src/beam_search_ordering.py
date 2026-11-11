import json
import csv

from pathlib import Path


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

SENTENCE_FILE = (
    ROOT / "data/processed/segmented/reviewed/"
    "sample-1_ordering_candidates.json"
)

SCORE_FILE = (
    ROOT / "data/processed/ordering_analysis/"
    "ranked_transitions.csv"
)

OUTPUT_DIR = (
    ROOT / "data/processed/ordering_analysis/"
    "beam_search_results"
)

OUTPUT_JSON = (
    OUTPUT_DIR / "beam_search_candidates.json"
)

OUTPUT_TXT = (
    OUTPUT_DIR / "best_ordered_candidate.txt"
)

BEAM_WIDTH = 25

TOP_RESULTS = 5


# ==========================================
# 2. LOAD SENTENCES
# ==========================================

def load_sentences():

    if not SENTENCE_FILE.exists():
        raise FileNotFoundError(SENTENCE_FILE)

    with open(
        SENTENCE_FILE,
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    sentence_map = {
        record["sentence_id"]: record["text"]
        for record in records
    }

    if not sentence_map:
        raise ValueError("No sentences found.")

    if len(sentence_map) != len(records):
        raise ValueError("Duplicate sentence IDs found.")

    return sentence_map


# ==========================================
# 3. LOAD TRANSITION SCORES
# ==========================================

def load_transition_scores():

    if not SCORE_FILE.exists():
        raise FileNotFoundError(SCORE_FILE)

    score_map = {}

    with open(
        SCORE_FILE,
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            source = int(row["sentence_a_id"])

            target = int(row["sentence_b_id"])

            score = float(row["transition_score"])

            if not -1 <= score <= 1:
                raise ValueError(
                    f"Unexpected transition score: {score}"
                )

            if (source, target) in score_map:
                raise ValueError(
                    f"Duplicate transition: {source}->{target}"
                )

            score_map[(source, target)] = score

    return score_map


# ==========================================
# 4. BEAM SEARCH FOR ONE STARTING SENTENCE
# ==========================================

def beam_search_for_start(
    start_id,
    sentence_ids,
    score_map,
    beam_width
):

    # Each path contains:
    # (cumulative_score, sequence)

    beam = [
        (0.0, (start_id,))
    ]

    total_sentences = len(sentence_ids)

    # Add one sentence at each iteration
    for step in range(total_sentences - 1):

        expanded_paths = []

        for cumulative_score, path in beam:

            used_sentences = set(path)

            current_sentence = path[-1]

            for next_sentence in sentence_ids:

                # Do not repeat sentences
                if next_sentence in used_sentences:
                    continue

                edge = (
                    current_sentence,
                    next_sentence
                )

                if edge not in score_map:
                    continue

                transition_score = score_map[edge]

                new_score = (
                    cumulative_score + transition_score
                )

                new_path = path + (next_sentence,)

                expanded_paths.append(
                    (new_score, new_path)
                )

        if not expanded_paths:
            break

        # Keep only the best K paths
        expanded_paths.sort(
            key=lambda item: item[0],
            reverse=True
        )

        beam = expanded_paths[:beam_width]

    # Return only complete sequences
    complete_paths = [
        item
        for item in beam
        if len(item[1]) == total_sentences
    ]

    return complete_paths


# ==========================================
# 5. SEARCH ALL POSSIBLE STARTING SENTENCES
# ==========================================

def find_best_sequences(
    sentence_ids,
    score_map
):

    all_candidates = []

    print("\nStarting Beam Search...")

    for index, start_id in enumerate(sentence_ids, start=1):

        print(
            f"Exploring start {index}/{len(sentence_ids)} "
            f"(S{start_id})"
        )

        candidates = beam_search_for_start(
            start_id,
            sentence_ids,
            score_map,
            BEAM_WIDTH
        )

        all_candidates.extend(candidates)

    # Remove duplicate sequences
    unique_candidates = {}

    for score, path in all_candidates:

        unique_candidates[path] = score

    ranked = sorted(
        unique_candidates.items(),
        key=lambda item: item[1],
        reverse=True
    )

    return ranked[:TOP_RESULTS]


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(
    candidates,
    sentence_map
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    report = []

    for rank, (path, score) in enumerate(
        candidates,
        start=1
    ):

        report.append({

            "rank": rank,

            "cumulative_score": round(score, 5),

            "mean_transition_score": round(
                score / (len(path) - 1), 5
            ),

            "sentence_ids": list(path),

            "sentences": [
                sentence_map[sid]
                for sid in path
            ]

        })

    # Save JSON
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Save best candidate as readable text
    if report:

        best = report[0]

        with open(
            OUTPUT_TXT,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                "BEST SCORING CANDIDATE - NOT VERIFIED\n\n"
            )

            for index, sentence in enumerate(
                best["sentences"],
                start=1
            ):

                sid = best["sentence_ids"][index - 1]

                file.write(
                    f"S{sid}: {sentence}\n\n"
                )

    print("\nResults saved successfully!")


# ==========================================
# 7. MAIN
# ==========================================

def main():

    sentence_map = load_sentences()

    sentence_ids = list(sentence_map.keys())

    score_map = load_transition_scores()

    expected_transitions = (
        len(sentence_ids) * (len(sentence_ids) - 1)
    )

    print("\n========== DATASET ==========")

    print("Sentences:", len(sentence_ids))

    print("Available transitions:", len(score_map))

    print("Expected transitions:", expected_transitions)

    if len(score_map) != expected_transitions:

        raise ValueError(
            "Transition count mismatch. "
            "Regenerate directed features and scores."
        )

    # Search
    candidates = find_best_sequences(
        sentence_ids,
        score_map
    )

    if not candidates:

        raise RuntimeError(
            "No complete sentence sequences found."
        )

    # Save
    save_results(
        candidates,
        sentence_map
    )

    # Display results
    print("\n========== TOP CANDIDATE SEQUENCES ==========")

    for rank, (path, score) in enumerate(
        candidates[:3],
        start=1
    ):

        print(f"\nRANK {rank}")

        print("Sentence IDs:", path)

        print(
            "Mean transition score:",
            round(score / (len(path) - 1), 5)
        )

        print("\nFirst five sentences:")

        for sid in path[:5]:

            print(f"S{sid}: {sentence_map[sid]}")

    print("\n========== COMPLETE ==========")

    print("JSON:", OUTPUT_JSON)

    print("Text:", OUTPUT_TXT)


if __name__ == "__main__":
    main()
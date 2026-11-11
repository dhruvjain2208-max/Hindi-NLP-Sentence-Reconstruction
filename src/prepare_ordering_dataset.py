import json
import csv

from pathlib import Path


# =====================================
# 1. PATH CONFIGURATION
# =====================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/segmented/"
    "sample-1_sentences.json"
)

OUTPUT_DIR = (
    ROOT / "data/processed/segmented/reviewed"
)

REVIEWED_FILE = (
    OUTPUT_DIR / "sample-1_sentences_reviewed.json"
)

CANDIDATES_FILE = (
    OUTPUT_DIR / "sample-1_ordering_candidates.json"
)

REVIEW_FILE = (
    OUTPUT_DIR / "sample-1_manual_review.csv"
)


# =====================================
# 2. REVIEW RULES
# =====================================

# Manually identified suspicious sentence
MANUAL_REVIEW_IDS = {43}

# Automatically flag sentences with 2 or fewer words
MIN_WORD_COUNT = 2


# =====================================
# 3. PROCESS SENTENCES
# =====================================

def prepare_dataset():

    if not INPUT_FILE.exists():
        raise FileNotFoundError(INPUT_FILE)

    records = json.loads(
        INPUT_FILE.read_text(encoding="utf-8")
    )

    reviewed_records = []
    ordering_candidates = []
    review_queue = []

    for record in records:

        sentence_id = record["sentence_id"]
        text = record["text"]

        word_count = len(text.split())

        reasons = []

        if word_count <= MIN_WORD_COUNT:
            reasons.append("Very short sentence")

        if sentence_id in MANUAL_REVIEW_IDS:
            reasons.append("Manually flagged fragment")

        # Preserve original sentence
        updated_record = {
            **record,
            "word_count": word_count,
            "review_status": (
                "needs_review" if reasons else "approved"
            ),
            "review_reason": reasons
        }

        reviewed_records.append(updated_record)

        if reasons:

            review_queue.append({
                "sentence_id": sentence_id,
                "text": text,
                "word_count": word_count,
                "reason": "; ".join(reasons),
                "decision": "pending"
            })

        else:

            ordering_candidates.append(record)

    # =================================
    # 4. SAVE OUTPUTS
    # =================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    REVIEWED_FILE.write_text(
        json.dumps(
            reviewed_records,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    CANDIDATES_FILE.write_text(
        json.dumps(
            ordering_candidates,
            ensure_ascii=False,
            indent=4
        ),
        encoding="utf-8"
    )

    with open(
        REVIEW_FILE,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=[
                "sentence_id",
                "text",
                "word_count",
                "reason",
                "decision"
            ]
        )

        writer.writeheader()
        writer.writerows(review_queue)

    # =================================
    # 5. DISPLAY RESULTS
    # =================================

    print("\n========== DATASET QUALITY REPORT ==========")

    print("Original sentences:", len(records))

    print("Approved candidates:", len(ordering_candidates))

    print("Sentences needing review:", len(review_queue))

    print("\nReview queue:")

    for item in review_queue:
        print(
            f"S{item['sentence_id']}: "
            f"{item['text']}"
        )

    print("\n========== OUTPUT FILES ==========")

    print("Reviewed:", REVIEWED_FILE)

    print("Ordering candidates:", CANDIDATES_FILE)

    print("Manual review:", REVIEW_FILE)


if __name__ == "__main__":
    prepare_dataset()
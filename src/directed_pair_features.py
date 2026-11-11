import json
import csv
import re
import torch

from pathlib import Path
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "data/processed/segmented/reviewed/"
    "sample-1_ordering_candidates.json"
)

OUTPUT_DIR = (
    ROOT / "data/processed/ordering_analysis"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "directed_pair_features.csv"
)

TOP_FILE = (
    OUTPUT_DIR / "top_successor_candidates.csv"
)

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

DEVICE = "cpu"

BATCH_SIZE = 16


# ==========================================
# 2. HINDI STOPWORDS
# ==========================================

STOPWORDS = {
    "के", "का", "की", "को", "से", "में",
    "पर", "और", "तथा", "है", "था", "थी",
    "थे", "हैं", "हो", "एक", "यह", "वह",
    "इस", "उस", "इन", "उन", "भी", "ही",
    "तो", "कि", "जो", "जब", "कुछ", "लिए",
    "द्वारा", "अपने", "अपना", "अपनी",
    "सभी", "बहुत", "या", "करने", "कर",
    "कहा", "जाता", "जाती", "रहा", "रही",
    "रहे", "हुए", "हुआ", "हुई"
}


# ==========================================
# 3. DISCOURSE MARKERS
# ==========================================

CONNECTIVES = [
    "इसलिए",
    "क्योंकि",
    "लेकिन",
    "किंतु",
    "अतः",
    "फलस्वरूप",
    "इसके बाद",
    "इसके पश्चात",
    "फिर",
    "तब",
    "तथा",
    "हालांकि"
]


REFERENCE_MARKERS = [
    "उक्त",
    "इसका",
    "उसका",
    "इसके",
    "उसके",
    "इनका",
    "उनका",
    "यह",
    "वह",
    "इस",
    "उस",
    "ऐसा",
    "ऐसी",
    "ऐसे",
    "वहाँ",
    "वहां",
    "तभी"
]


TEMPORAL_MARKERS = [
    "इसके बाद",
    "इसके पश्चात",
    "उस समय",
    "बाद में",
    "तब",
    "फिर",
    "पहले",
    "अंततः",
    "अब",
    "उस वक्त"
]


# ==========================================
# 4. LOAD MODEL
# ==========================================

print("Loading MiniLM tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading MiniLM model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
).to(DEVICE)

model.eval()

print("Model loaded successfully!")


# ==========================================
# 5. TEXT TOKENIZATION
# ==========================================

def tokenize_text(text):

    tokens = re.findall(
        r"[^\W_]+",
        text.lower(),
        flags=re.UNICODE
    )

    return [
        token for token in tokens
        if token != "null"
    ]


def get_content_words(text):

    return {
        token
        for token in tokenize_text(text)
        if token not in STOPWORDS
        and not token.isdigit()
    }


def get_numbers(text):

    return set(
        re.findall(r"\d+", text)
    )


def get_latin_terms(text):

    return {
        word.lower()
        for word in re.findall(
            r"[A-Za-z]+",
            text
        )
        if word.upper() != "NULL"
    }


# ==========================================
# 6. MEAN POOLING
# ==========================================

def mean_pooling(output, attention_mask):

    embeddings = output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        embeddings.size()
    ).float()

    summed = torch.sum(
        embeddings * mask,
        dim=1
    )

    counts = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return summed / counts


# ==========================================
# 7. GENERATE EMBEDDINGS
# ==========================================

def generate_embeddings(sentences):

    all_embeddings = []

    for start in range(0, len(sentences), BATCH_SIZE):

        batch = sentences[
            start:start + BATCH_SIZE
        ]

        # Ignore NULL placeholders for semantic
        # embedding, but preserve original text.
        clean_batch = [
            re.sub(
                r"\bNULL\b",
                " ",
                sentence,
                flags=re.IGNORECASE
            )
            for sentence in batch
        ]

        inputs = tokenizer(
            clean_batch,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.inference_mode():

            output = model(**inputs)

            embeddings = mean_pooling(
                output,
                inputs["attention_mask"]
            )

            embeddings = F.normalize(
                embeddings,
                p=2,
                dim=1
            )

        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0)


# ==========================================
# 8. DIRECTIONAL FEATURES
# ==========================================

def starts_with_marker(sentence, markers):

    text = sentence.strip().lower()

    return int(
        any(
            text.startswith(marker)
            for marker in markers
        )
    )


def jaccard_similarity(words_a, words_b):

    union = words_a | words_b

    if not union:
        return 0.0

    intersection = words_a & words_b

    return len(intersection) / len(union)


def extract_features(
    sentence_a,
    sentence_b,
    embedding_a,
    embedding_b
):

    words_a = get_content_words(sentence_a)

    words_b = get_content_words(sentence_b)

    # Semantic similarity
    semantic_similarity = torch.dot(
        embedding_a,
        embedding_b
    ).item()

    # Shared content vocabulary
    shared_words = words_a & words_b

    content_overlap = jaccard_similarity(
        words_a,
        words_b
    )

    # First six meaningful tokens in B
    leading_words_b = [
        word
        for word in tokenize_text(sentence_b)
        if word not in STOPWORDS
        and not word.isdigit()
    ][:6]

    leading_overlap = len(
        words_a & set(leading_words_b)
    )

    # Numerical continuity
    numbers_a = get_numbers(sentence_a)

    numbers_b = get_numbers(sentence_b)

    shared_numbers = numbers_a & numbers_b

    # English terminology overlap
    latin_a = get_latin_terms(sentence_a)

    latin_b = get_latin_terms(sentence_b)

    shared_latin = latin_a & latin_b

    return {

        "semantic_similarity":
            semantic_similarity,

        "content_jaccard":
            content_overlap,

        "shared_content_count":
            len(shared_words),

        "leading_overlap_count":
            leading_overlap,

        "b_starts_with_connective":
            starts_with_marker(
                sentence_b,
                CONNECTIVES
            ),

        "b_starts_with_reference":
            starts_with_marker(
                sentence_b,
                REFERENCE_MARKERS
            ),

        "b_starts_with_temporal":
            starts_with_marker(
                sentence_b,
                TEMPORAL_MARKERS
            ),

        "shared_number_count":
            len(shared_numbers),

        "shared_latin_count":
            len(shared_latin)

    }


# ==========================================
# 9. MAIN FEATURE EXTRACTION
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"{INPUT_FILE} not found.\n"
            "Run prepare_ordering_dataset.py first."
        )

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as file:

        records = json.load(file)

    sentences = [
        record["text"]
        for record in records
    ]

    sentence_ids = [
        record["sentence_id"]
        for record in records
    ]

    print("\nTotal sentences:", len(sentences))

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(sentences)

    print("Embedding shape:", embeddings.shape)

    rows = []

    # Generate all directed pairs
    for i in range(len(sentences)):

        for j in range(len(sentences)):

            if i == j:
                continue

            features = extract_features(
                sentences[i],
                sentences[j],
                embeddings[i],
                embeddings[j]
            )

            row = {

                "sentence_a_id": sentence_ids[i],

                "sentence_b_id": sentence_ids[j],

                "sentence_a": sentences[i],

                "sentence_b": sentences[j],

                **features

            }

            rows.append(row)

    # Save directed features
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

    print("\n========== FEATURE EXTRACTION ==========")

    print("Directed pairs:", len(rows))

    print("Features per pair:",
          len(rows[0]) - 4)

    print("Saved:", OUTPUT_FILE)

    # ----------------------------------
    # Display top similarity candidates
    # ----------------------------------

    top_candidates = []

    for sid in sentence_ids:

        candidates = [
            row for row in rows
            if row["sentence_a_id"] == sid
        ]

        candidates.sort(
            key=lambda row: row["semantic_similarity"],
            reverse=True
        )

        top_candidates.extend(
            candidates[:3]
        )

    with open(
        TOP_FILE,
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

    print("Top candidates saved:", TOP_FILE)

    print("\n========== COMPLETE ==========")


if __name__ == "__main__":

    main()
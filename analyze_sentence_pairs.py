import torch
import torch.nn.functional as F
import json
import csv

from pathlib import Path

from transformers import AutoTokenizer, AutoModel


# =====================================
# 1. CONFIGURATION
# =====================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

INPUT_FILE = Path(
    "data/processed/segmented/sample-1_sentences.json"
)

OUTPUT_DIR = Path(
    "data/processed/ordering_analysis"
)

DEVICE = "cpu"


# =====================================
# 2. LOAD MODEL
# =====================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# =====================================
# 3. MEAN POOLING
# =====================================

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    sum_embeddings = torch.sum(
        token_embeddings * mask,
        dim=1
    )

    sum_mask = torch.clamp(
        mask.sum(dim=1),
        min=1e-9
    )

    return sum_embeddings / sum_mask


# =====================================
# 4. GENERATE EMBEDDINGS
# =====================================

def generate_embeddings(sentences):

    encoded = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    with torch.inference_mode():

        output = model(**encoded)

        embeddings = mean_pooling(
            output,
            encoded["attention_mask"]
        )

        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

    return embeddings


# =====================================
# 5. LOAD SEGMENTED SENTENCES
# =====================================

def load_sentences():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Missing segmented file: {INPUT_FILE}"
        )

    records = json.loads(
        INPUT_FILE.read_text(encoding="utf-8")
    )

    sentences = [
        record["text"]
        for record in records
        if record.get("text", "").strip()
    ]

    return sentences


# =====================================
# 6. CALCULATE SIMILARITY MATRIX
# =====================================

def calculate_similarity(embeddings):

    return torch.mm(
        embeddings,
        embeddings.T
    )


# =====================================
# 7. SAVE ALL PAIR SCORES
# =====================================

def save_pair_scores(sentences, matrix):

    path = OUTPUT_DIR / "pairwise_similarity.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "sentence_a_id",
            "sentence_b_id",
            "similarity"
        ])

        n = len(sentences)

        for i in range(n):

            for j in range(i + 1, n):

                writer.writerow([
                    i + 1,
                    j + 1,
                    round(matrix[i][j].item(), 4)
                ])

    print("Saved pair scores:", path)


# =====================================
# 8. FIND TOP RELATED SENTENCES
# =====================================

def find_top_neighbors(sentences, matrix, top_k=3):

    path = OUTPUT_DIR / "top_neighbors.csv"

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "sentence_id",
            "sentence",
            "candidate_id",
            "candidate_sentence",
            "similarity"
        ])

        for i in range(len(sentences)):

            candidates = []

            for j in range(len(sentences)):

                if i == j:
                    continue

                score = matrix[i][j].item()

                candidates.append((score, j))

            candidates.sort(reverse=True)

            for score, j in candidates[:top_k]:

                writer.writerow([
                    i + 1,
                    sentences[i],
                    j + 1,
                    sentences[j],
                    round(score, 4)
                ])

    print("Saved nearest neighbors:", path)


# =====================================
# 9. MAIN
# =====================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    sentences = load_sentences()

    print("\nTotal sentences:", len(sentences))

    print("\nGenerating embeddings...")

    embeddings = generate_embeddings(sentences)

    print("Embedding shape:", embeddings.shape)

    print("\nCalculating similarity matrix...")

    matrix = calculate_similarity(embeddings)

    save_pair_scores(sentences, matrix)

    find_top_neighbors(sentences, matrix)

    print("\n========== SAMPLE RESULTS ==========")

    for i in range(min(5, len(sentences))):

        print(f"\nSentence {i + 1}:")
        print(sentences[i])

        candidates = []

        for j in range(len(sentences)):

            if i != j:

                candidates.append((
                    matrix[i][j].item(),
                    j
                ))

        candidates.sort(reverse=True)

        print("\nTop 3 related sentences:")

        for score, j in candidates[:3]:

            print(
                f"\nS{j + 1} | Similarity: {score:.4f}"
            )

            print(sentences[j])


if __name__ == "__main__":

    main()
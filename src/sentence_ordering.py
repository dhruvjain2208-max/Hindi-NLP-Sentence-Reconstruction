import torch
import torch.nn.functional as F

from transformers import AutoTokenizer, AutoModel


# ==========================================
# 1. CONFIGURATION
# ==========================================

MODEL_NAME = (
    "sentence-transformers/"
    "paraphrase-multilingual-MiniLM-L12-v2"
)

DEVICE = "cpu"


# ==========================================
# 2. LOAD TOKENIZER
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ==========================================
# 3. LOAD PRETRAINED MODEL
# ==========================================

print("Loading pretrained model...")

model = AutoModel.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)

model.eval()


print("\nModel loaded successfully!")

print("Device:", DEVICE)


# ==========================================
# 4. MEAN POOLING
# ==========================================

def mean_pooling(model_output, attention_mask):

    token_embeddings = model_output.last_hidden_state

    input_mask = attention_mask.unsqueeze(-1).expand(
        token_embeddings.size()
    ).float()

    sum_embeddings = torch.sum(
        token_embeddings * input_mask,
        dim=1
    )

    sum_mask = torch.clamp(
        input_mask.sum(dim=1),
        min=1e-9
    )

    return sum_embeddings / sum_mask


# ==========================================
# 5. GENERATE SENTENCE EMBEDDINGS
# ==========================================

def generate_embeddings(sentences):

    inputs = tokenizer(
        sentences,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        outputs = model(**inputs)

        embeddings = mean_pooling(
            outputs,
            inputs["attention_mask"]
        )

        # Normalize vectors
        embeddings = F.normalize(
            embeddings,
            p=2,
            dim=1
        )

    return embeddings


# ==========================================
# 6. TEST
# ==========================================

if __name__ == "__main__":

    sentences = [

        "वह बचपन से ही विज्ञान में रुचि रखता था।",

        "उसने वैज्ञानिक विचारों पर प्रयोग करना शुरू किया।",

        "आज मौसम बहुत अच्छा है।"

    ]

    embeddings = generate_embeddings(sentences)

    print("\n========== EMBEDDINGS ==========")

    print("Embedding shape:", embeddings.shape)

    for i, embedding in enumerate(embeddings):

        print(f"\nSentence {i + 1}:")

        print(sentences[i])

        print("Vector dimensions:", len(embedding))


    # ======================================
    # 7. COSINE SIMILARITY
    # ======================================

    similarity_matrix = torch.mm(
        embeddings,
        embeddings.T
    )

    print("\n========== SIMILARITY MATRIX ==========")

    print(similarity_matrix)
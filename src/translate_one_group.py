import json
import torch

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# ==========================================
# 1. CONFIGURATION
# ==========================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

INPUT_FILE = (
    ROOT / "data/processed/translation_input/"
    "sample-1_groups.json"
)

OUTPUT_DIR = (
    ROOT / "data/processed/translations"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "sample-1_group-01_translation.json"
)

DEVICE = "cpu"


# ==========================================
# 2. LOAD PRETRAINED MODEL
# ==========================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

print("Loading translation model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Translation model loaded successfully!")

print("Device:", DEVICE)


# ==========================================
# 3. TRANSLATION FUNCTION
# ==========================================

def translate_text(hindi_text):

    inputs = tokenizer(
        hindi_text,
        return_tensors="pt",
        truncation=False
    )

    # Check model input limit instead of
    # silently truncating the paragraph.

    maximum_length = getattr(
        model.config,
        "max_position_embeddings",
        512
    )

    input_length = inputs["input_ids"].shape[1]

    print("Input tokens:", input_length)

    print("Maximum model positions:", maximum_length)

    if input_length > maximum_length:

        raise ValueError(
            "The paragraph exceeds the model's "
            "input length. Do not silently truncate it."
        )

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=4,
            early_stopping=True
        )

    english_text = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return english_text[0]


# ==========================================
# 4. LOAD FIRST GROUP
# ==========================================

def load_first_group():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Grouping file missing: {INPUT_FILE}\n"
            "Run group_sentences.py first."
        )

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    groups = data["groups"]

    if not groups:

        raise ValueError("No sentence groups found.")

    first_group = groups[0]

    sentences = first_group["sentences"]

    hindi_paragraph = " ".join(
        sentence["text"]
        for sentence in sentences
    )

    sentence_ids = [
        sentence["sentence_id"]
        for sentence in sentences
    ]

    return (
        first_group["group_id"],
        sentence_ids,
        hindi_paragraph
    )


# ==========================================
# 5. MAIN
# ==========================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    group_id, sentence_ids, hindi_text = (
        load_first_group()
    )

    print("\n========== HINDI INPUT ==========")

    print(hindi_text)

    print("\nSentence IDs:", sentence_ids)

    print("\nTranslating paragraph...")

    english_text = translate_text(hindi_text)

    print("\n========== ENGLISH TRANSLATION ==========")

    print(english_text)

    # Save structured output

    result = {

        "document": "sample-1",

        "group_id": group_id,

        "sentence_ids": sentence_ids,

        "model": MODEL_NAME,

        "device": DEVICE,

        "ordering_status": "UNVERIFIED_CANDIDATE",

        "source_language": "Hindi",

        "target_language": "English",

        "hindi_text": hindi_text,

        "english_translation": english_text

    }

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\nTranslation completed!")

    print("Output saved:", OUTPUT_FILE)


if __name__ == "__main__":

    main()
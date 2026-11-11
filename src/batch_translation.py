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
    ROOT / "data/processed/translations/sample-1"
)

OUTPUT_JSON = OUTPUT_DIR / "all_translations.json"

OUTPUT_TXT = OUTPUT_DIR / "translated_document.txt"

DEVICE = "cpu"

BEAM_SIZE = 4


# ==========================================
# 2. LOAD MODEL
# ==========================================

def load_model():

    print("Loading OPUS-MT tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME
    )

    print("Loading OPUS-MT model...")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME
    )

    model.to(DEVICE)
    model.eval()

    print("Translation model loaded successfully!")

    return tokenizer, model


# ==========================================
# 3. TRANSLATION FUNCTION
# ==========================================

def translate_text(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=False
    )

    input_length = inputs["input_ids"].shape[1]

    model_limit = getattr(
        model.config,
        "max_position_embeddings",
        512
    )

    if not isinstance(model_limit, int) or model_limit <= 0:
        model_limit = 512

    if input_length > model_limit:

        raise ValueError(
            f"Input contains {input_length} tokens, "
            f"but model limit is {model_limit}. "
            "Split the paragraph rather than truncating it."
        )

    with torch.inference_mode():

        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            num_beams=BEAM_SIZE,
            early_stopping=True
        )

    translation = tokenizer.batch_decode(
        outputs,
        skip_special_tokens=True
    )

    return translation[0]


# ==========================================
# 4. LOAD GROUPS
# ==========================================

def load_groups():

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Input file missing: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data


# ==========================================
# 5. TRANSLATE ALL GROUPS
# ==========================================

def process_groups(data, tokenizer, model):

    results = []

    groups = data["groups"]

    print("\nTotal groups:", len(groups))

    for group in groups:

        group_id = group["group_id"]

        sentences = group["sentences"]

        hindi_text = " ".join(
            sentence["text"]
            for sentence in sentences
        )

        sentence_ids = [
            sentence["sentence_id"]
            for sentence in sentences
        ]

        print("\n" + "=" * 55)

        print(f"TRANSLATING GROUP {group_id}")

        print("Sentence IDs:", sentence_ids)

        print("\nHindi:")
        print(hindi_text)

        try:

            english_text = translate_text(
                hindi_text,
                tokenizer,
                model
            )

            print("\nEnglish:")
            print(english_text)

            results.append({

                "group_id": group_id,

                "sentence_ids": sentence_ids,

                "sentence_count": len(sentences),

                "hindi_text": hindi_text,

                "english_translation": english_text,

                "translation_status": "completed"

            })

        except Exception as error:

            print("\nTranslation failed:", error)

            results.append({

                "group_id": group_id,

                "sentence_ids": sentence_ids,

                "hindi_text": hindi_text,

                "english_translation": None,

                "translation_status": "failed",

                "error": str(error)

            })

    return results


# ==========================================
# 6. SAVE RESULTS
# ==========================================

def save_results(results, original_data):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    completed = sum(
        1 for result in results
        if result["translation_status"] == "completed"
    )

    output_data = {

        "document": "sample-1",

        "translation_model": MODEL_NAME,

        "device": DEVICE,

        "ordering_status": original_data.get(
            "ordering_status",
            "UNVERIFIED_CANDIDATE"
        ),

        "total_groups": len(results),

        "completed_groups": completed,

        "groups": results

    }

    # Save structured JSON
    with open(
        OUTPUT_JSON,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # Save readable document
    with open(
        OUTPUT_TXT,
        "w",
        encoding="utf-8"
    ) as file:

        for result in results:

            file.write(
                f"\nGROUP {result['group_id']}\n"
            )

            file.write("-" * 50 + "\n")

            file.write("HINDI:\n")

            file.write(
                result["hindi_text"] + "\n\n"
            )

            file.write("ENGLISH:\n")

            file.write(
                str(result["english_translation"])
                + "\n\n"
            )

    print("\n========== TRANSLATION SUMMARY ==========")

    print("Total groups:", len(results))

    print("Completed:", completed)

    print("Failed:", len(results) - completed)

    print("\nJSON:", OUTPUT_JSON)

    print("Text:", OUTPUT_TXT)


# ==========================================
# 7. MAIN
# ==========================================

def main():

    data = load_groups()

    tokenizer, model = load_model()

    results = process_groups(
        data,
        tokenizer,
        model
    )

    save_results(
        results,
        data
    )

    print("\nBATCH TRANSLATION FINISHED!")


if __name__ == "__main__":

    main()
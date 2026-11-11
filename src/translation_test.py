import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)


# =================================
# 1. CONFIGURATION
# =================================

MODEL_NAME = "Helsinki-NLP/opus-mt-hi-en"

DEVICE = "cpu"


# =================================
# 2. LOAD TOKENIZER
# =================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# =================================
# 3. LOAD MODEL
# =================================

print("Loading translation model...")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully!")


# =================================
# 4. TRANSLATION FUNCTION
# =================================

def translate(text):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )

    with torch.inference_mode():

        output = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4
        )

    translation = tokenizer.batch_decode(
        output,
        skip_special_tokens=True
    )

    return translation[0]


# =================================
# 5. TEST
# =================================

hindi_text = (
    "हिंदू धर्म में तीर्थ का बड़ा महत्व है। "
    "गढ़मुक्तेश्वर हिंदुओं का पावन तीर्थ है।"
)

print("\nOriginal Hindi:")
print(hindi_text)

english_text = translate(hindi_text)

print("\nEnglish Translation:")
print(english_text)
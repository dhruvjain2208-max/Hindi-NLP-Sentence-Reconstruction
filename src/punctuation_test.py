from pathlib import Path
import importlib.util

import torch
from huggingface_hub import hf_hub_download


# ---------------------------------
# 1. Model configuration
# ---------------------------------

MODEL_ID = "Horizon-Labs/punctuation-restoration-small"

INPUT_FILE = Path(
    "data/processed/normalized/"
    "Sample-Raw-No-Punctuations/sample-1.txt"
)

OUTPUT_FILE = Path(
    "data/processed/punctuation_preview/sample-1.txt"
)


# ---------------------------------
# 2. Download punctuation helper
# ---------------------------------

print("Downloading model helper...")

helper_path = hf_hub_download(
    repo_id=MODEL_ID,
    filename="punctuate.py"
)


# ---------------------------------
# 3. Import helper
# ---------------------------------

spec = importlib.util.spec_from_file_location(
    "punctuate_helper",
    helper_path
)

punctuate_module = importlib.util.module_from_spec(spec)

spec.loader.exec_module(punctuate_module)


# ---------------------------------
# 4. Load pretrained model
# ---------------------------------

print("Loading pretrained model...")

punctuator = punctuate_module.Punctuator(
    MODEL_ID,
    device="cpu"
)

print("Model loaded successfully!")

print(
    "Model device:",
    next(punctuator.model.parameters()).device
)


# ---------------------------------
# 5. Read input text
# ---------------------------------

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

original_text = INPUT_FILE.read_text(
    encoding="utf-8"
)


# ---------------------------------
# 6. Restore punctuation
# ---------------------------------

print("\nRestoring punctuation...")

punctuated_text = punctuator(original_text)


# ---------------------------------
# 7. Save output
# ---------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE.write_text(
    punctuated_text,
    encoding="utf-8"
)


# ---------------------------------
# 8. Display results
# ---------------------------------

print("\n========== ORIGINAL TEXT ==========\n")

print(original_text[:1000])

print("\n========== PUNCTUATED TEXT ==========\n")

print(punctuated_text[:1000])

print("\n========== COMPLETE ==========")

print("Output saved to:", OUTPUT_FILE)
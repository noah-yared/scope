from pathlib import Path
import os
import sys
import pandas as pd
import json
import re
from transformers import AutoTokenizer
from typing import Any

# add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.problem_mappings import ProblemType, PROBLEM_TYPES, PROBLEM_MAPPING

# Default tokenizer is GPT-2 BPE tokenizer
DEFAULT_MODEL_ID = "gpt2"

# Cache used tokenizers in memory for quick retrieval
TOKENIZER_CACHE: dict[str, Any] = {}


def retrieve_tokenizer(model_id: str) -> Any:
    """
    Search for loaded tokenizer in cache.
    If not present, load the tokenizer for
    model_id from huggingface.
    """
    if model_id in TOKENIZER_CACHE:
        return TOKENIZER_CACHE[model_id]
    tokenizer = AutoTokenizer.from_pretrained(model_id) 
    TOKENIZER_CACHE[model_id] = tokenizer
    return tokenizer


def compute_token_count(text: str, model_id: str) -> int:
    """
    Compute the number of tokens in the given text using the tokenizer
    for model_id.
    """
    # get tokenizer for model_id
    tokenizer = retrieve_tokenizer(model_id)

    # encode prompt, output and useful output
    encoded_text = tokenizer.encode(text, add_special_tokens=True)

    return len(encoded_text)


def display_schema_token_counts(model_id: str, schema_dir: Path) -> None:
    token_counts = {}
    for schema_file in os.listdir(schema_dir):
        if re.match(r".*_schema\.txt", schema_file) is None:
            # not a schema file
            continue
        with open(schema_dir / schema_file, "r") as f:
            schema_str = f.read()
        # extract problem category
        problem_category = re.match(r"(\w*)_schema\.txt", schema_file).group(1)
        token_counts[problem_category] = compute_token_count(schema_str, model_id)

    # pretty print schema token counts per problem category
    print()
    print("############################")
    print("### SCHEMA TOKEN LENGTHS ###")
    print("############################")
    print(json.dumps(token_counts, indent=2))

    # display average token counts across schemas
    average_token_length = sum(token_counts.values()) / len(token_counts)
    print("\nAverage token length:", average_token_length, end="\n\n")


if __name__ == "__main__":
    model_id = DEFAULT_MODEL_ID
    schema_dir = Path(__file__).parent.parent / "old_schemas"
    display_schema_token_counts(model_id, schema_dir)

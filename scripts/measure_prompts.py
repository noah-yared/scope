from pathlib import Path
import os
import sys
import pandas as pd
import json
import re
from collections import defaultdict
from transformers import AutoTokenizer
from typing import Any

# add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.problem_mappings import ProblemType, PROBLEM_TYPES, PROBLEM_MAPPING
from scripts.prepare_clrs_dataset import prepare_clrs_dataset
from scripts.prompt_templates import BASE_PROMPT, COT_PROMPT, REACT_PROMPT, SCOPE_PROMPT

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


def compute_token_count(text: str, model_id: str = "gpt2") -> int:
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
    token_counts = defaultdict(int)
    for schema_file in os.listdir(schema_dir):
        if re.match(r".*_(schema|example)\.txt", schema_file) is None:
            # not a schema/example file
            continue
        with open(schema_dir / schema_file, "r") as f:
            schema_str = f.read()
        # extract problem category
        problem_category = re.match(r"(\w*)_(schema|example)\.txt", schema_file).group(1)
        token_counts[problem_category] += compute_token_count(schema_str, model_id)

    # pretty print schema token counts per problem category
    print()
    print("############################")
    print("### SCHEMA TOKEN LENGTHS ###")
    print("############################")
    print(json.dumps(token_counts, indent=2))
    average_token_length = round(sum(token_counts.values()) / len(token_counts), 2)
    print(f"Average: {average_token_length} tokens")


def compute_avg_question_token_counts(df: pd.DataFrame) -> int:
    def trim_question(question: str) -> str:
        cleaned_question = re.sub(r"trace \| .*?", "", question)
        cleaned_question = re.sub(r"initial_trace: \[.*\]\n", "", cleaned_question)
        return cleaned_question.strip()
    return round(sum((map(
        lambda q: compute_token_count(q, DEFAULT_MODEL_ID),
        df['question'].map(trim_question)
    ))) / len(df), 2)


def compute_template_token_counts() -> dict[str, int]:
    return dict(zip(
        ["base", "cot", "react", "scope"],
        map(
            compute_token_count,
            [BASE_PROMPT, COT_PROMPT, REACT_PROMPT, SCOPE_PROMPT]
        )
    ))


def display_template_token_counts() -> None:
    print()
    print("############################")
    print("### PROMPT TOKEN LENGTHS ###")
    print("############################")
    print(json.dumps(compute_template_token_counts(), indent=2))


def display_avg_question_token_counts(dataset_file: Path) -> None:
    if not os.path.exists(dataset_file):
        fetch_counts = dict([(category, 100) for category in PROBLEM_TYPES])
        prepare_clrs_dataset(fetch_counts, sum(fetch_counts.values()))

    df = pd.read_parquet(dataset_file)

    print()
    print("#####################################")
    print("### AVERAGE QUESTION TOKEN LENGTH ###")
    print("#####################################")
    print(f"{compute_avg_question_token_counts(df)} tokens on average ({len(df)} total questions)")
    print()


def main() -> None:
    model_id = DEFAULT_MODEL_ID
    schema_dir = project_root / "old_schemas"
    dataset_file = project_root / "source_datasets" / "processed_clrs_dataset.parquet"
    display_schema_token_counts(model_id, schema_dir)
    display_template_token_counts()
    display_avg_question_token_counts(dataset_file)


if __name__ == "__main__":
    main()

import json
import pandas as pd
from pathlib import Path
import random
import re
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.prompt_templates import BASE_PROMPT, COT_PROMPT, REACT_PROMPT, SCOPE_PROMPT
from scripts.problem_mappings import ProblemType

BENCH_DIR = project_root / "benchmark_datasets"
SOURCE_DATASET = project_root / "source_datasets" / "processed_clrs_dataset.parquet"

# Compile regex patterns to speed up regex matches, since we do a lot
TRACE_PATTERN = re.compile(r"trace \| .*?")
INITIAL_TRACE_PATTERN = re.compile(r"initial_trace: \[.*\]\n")
# ANSWER_PATTERN = re.compile(r"(?<= \| ).*")
ANSWER_PATTERN = re.compile(r".*?\| ")

def fetch_example_outputs(question_df: pd.DataFrame, algorithm: str) -> tuple[str, str]:
    question_df = question_df[question_df['algorithm'] == algorithm]
    random_indices = random.sample(list(range(len(question_df))), k=2)
    example_output_A = question_df.iloc[random_indices[0]]['answer']
    example_output_B = question_df.iloc[random_indices[1]]['answer']
    return example_output_A, example_output_B


def process_questions(question_df: pd.DataFrame) -> pd.DataFrame:
    def trim_question(question: str) -> str:
        # remove trace and initial trace from clrs dataset questions
        trimmed_question = TRACE_PATTERN.sub("", question)
        trimmed_question = INITIAL_TRACE_PATTERN.sub("", trimmed_question)
        return trimmed_question.strip()
    def extract_answer(answer: str) -> str:
        trimmed_answer = ANSWER_PATTERN.sub("", answer)
        return trimmed_answer.strip()
    question_df['question'] = question_df['question'].map(trim_question)
    question_df['answer'] = question_df['answer'].map(extract_answer)
    return question_df


def make_non_scope_benchmark(method: str, question_df: pd.DataFrame, num_prompts: int) -> list[str]:
    prompt_template = {
        "cot": COT_PROMPT,
        "react": REACT_PROMPT,
        "base": BASE_PROMPT,
    }[method]

    dataset = []
    for i in range(min(len(question_df), num_prompts)):
        row = question_df.iloc[i]
        example_output_A, example_output_B = fetch_example_outputs(question_df, row['algorithm'])
        dataset.append({
            "algorithm": row['algorithm'],
            "category": row['category'],
            "prompt": prompt_template.format(
                algorithm_name=row['algorithm'],
                question=row['question'],
                example_output_A=example_output_A,
                example_output_B=example_output_B,
            ),
            "question": row['question'],
            "answer": row['answer']
        })
    return dataset


def make_scope_benchmark(question_df: pd.DataFrame, num_prompts: int) -> list[str]:
    def read_schema(category: str) -> Path:
        schema_file = Path(__file__).parent.parent / "old_schemas" / f"{category}_schema.txt"
        with open(schema_file, "r") as f:
            return f.read()

    def read_example(category: str) -> str:
        example_file = Path(__file__).parent.parent / "old_schemas" / f"{category}_example.txt"
        with open(example_file, "r") as f:
            return f.read()
    
    dataset = []
    for i in range(min(len(question_df), num_prompts)):
        row = question_df.iloc[i]
        schema = read_schema(row['category'])
        example = read_example(row['category'])
        example_output_A, example_output_B = fetch_example_outputs(question_df, row['algorithm'])
        dataset.append({
            "algorithm": row['algorithm'],
            "category": row['category'],
            "prompt": SCOPE_PROMPT.format(
                algorithm_name=f"Algorithm {i+1}",
                question=row['question'],
                example_output_A=example_output_A,
                example_output_B=example_output_B,
                worked_example=example,
                algorithm_schema=schema,
            ),
            "question": row['question'],
            "answer": row['answer']
        })

    return dataset


def ensure_dir(dir: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    return dir 


def make_benchmarks(question_df: pd.DataFrame, num_prompts: int) -> list[str]:
    question_df = process_questions(question_df)
    for method in ['cot', 'react', 'base', 'scope']:
        if method != 'scope':
            dataset = make_non_scope_benchmark(method, question_df, num_prompts)
        else:
            dataset = make_scope_benchmark(question_df, num_prompts)
        with open(ensure_dir(BENCH_DIR) / f"benchmark_{method}_{num_prompts}.json", "w") as f:
            json.dump(dataset, f, indent=2)


def main(dataset_path: Path = SOURCE_DATASET, num_prompts: int = 100):
    print("Making benchmarks...")
    question_df = pd.read_parquet(dataset_path)
    make_benchmarks(question_df, num_prompts)
    print("Benchmarks made successfully!")


if __name__ == "__main__":
    main()

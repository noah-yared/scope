from dataclasses import dataclass
from huggingface_hub import snapshot_download
import os
import pandas as pd
from pathlib import Path
import random
import sys
from types import MappingProxyType
import json

# add project root to Python path to allow imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.problem_mappings import ProblemType, PROBLEM_MAPPING, PROBLEM_TYPES

CLRS_TEXT_TRAIN_REPO = 'tomg-group-umd/CLRS-Text-train'
SAVE_DIR = project_root / "source_datasets"

def download_hf_dataset(hf_repo_id: str) -> str:
    return snapshot_download(repo_id=hf_repo_id, repo_type='dataset', allow_patterns='*.parquet')


def get_parquet_filepaths(root: str) -> list[Path]:
    data_path = Path(root) / 'data'
    contents = os.listdir(data_path)
    return [
        data_path / entry
        for entry in contents
        if entry.lower().endswith('.parquet')
    ]


def parse_files_to_df(files: list[Path]) -> pd.DataFrame:
    assert all(f.lower().endswith('.parquet') for f in map(str, files)), "Must be a .parquet file!"

    frames = [pd.read_parquet(f) for f in files]
    return pd.concat(frames, ignore_index=True, sort=False)


def filter_df(df: pd.DataFrame, fetch_counts: dict[ProblemType, int]) -> pd.DataFrame:
    def filter_ptype(pt: ProblemType) -> pd.DataFrame:
        ptype_df = df[df["category"] == pt]
        return ptype_df.reset_index(drop=True).loc[:fetch_counts.get(pt, 0)]
    split_frames = list(map(filter_ptype, PROBLEM_TYPES))
    filtered_df = pd.concat(split_frames, ignore_index=True)
    return filtered_df


def trim_df(df: pd.DataFrame, num_rows: int) -> pd.DataFrame:
    if len(df) <= num_rows:
        # no need to trim
        return df

    # fetch up to num_rows random rows from full_frame
    random_indices = random.sample(range(len(df)), k=min(num_rows, len(df)))
    truncated_frame = df.loc[random_indices]

    # reset indices so that they are consecutive
    truncated_frame.reset_index(inplace=True, drop=True) 

    return truncated_frame


def map_problem_types(df: pd.DataFrame) -> pd.DataFrame:
    new_df = df.copy()
    new_df.rename(columns={'algo_name': 'algorithm'}, inplace=True)
    new_df['category'] = new_df['algorithm'].map(PROBLEM_MAPPING)
    return new_df


def ensure_save_path(save_path: Path) -> Path:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return save_path


def prepare_clrs_dataset(fetch_counts: dict[ProblemType, int], num_rows: int) -> pd.DataFrame:
    """
    Fetch and process CLRS-Text dataset by mapping 30 original
    problem types ('algo_name' column) according to PROBLEM_MAPPING
    and keep up to `num_rows` rows. Fetches as many of each problem type
    pt without exceeding {fetch_counts.get(pt, 0)}. Return pandas 
    dataframe for resulting processed dataset.
    """
    parquet_files = get_parquet_filepaths(download_hf_dataset(CLRS_TEXT_TRAIN_REPO))
    mapped_df = map_problem_types(parse_files_to_df(parquet_files))
    prepared_df = trim_df(filter_df(mapped_df, fetch_counts), num_rows)

    prepared_df.to_parquet(ensure_save_path(SAVE_DIR / "processed_clrs_dataset.parquet"))

    return prepared_df


def print_usage() -> None:
    usage = """
python3 prepare_clrs_dataset.py --num-rows | -n <num_rows> [--fetch-counts-path | -f <fetch_path>]

    Arguments:
    --num-rows | -n <num_rows> - Maximum number of examples to fetch from the dataset.
    --fetch-counts-file | -f <fetch_file> - Path to JSON file containing maximum number of 
    examples to fetch for each problem type. See below for format:
    {
        "geometry": <num_examples>,
        "divide_and_conquer": <num_examples>,
        "greedy": <num_examples>,
        "dynamic_programming": <num_examples>,
        "graphs": <num_examples>,
        "strings": <num_examples>,
        "sorting": <num_examples>,
        "search": <num_examples>,
    },
    where <num_examples> is the maximum number of examples to fetch for each problem type.

    NOTE: If no fetch counts file is provided, we will default to fetching {num_rows // len(PROBLEM_TYPES)} examples
    for each problem type.

    Example usage:
    python3 prepare_clrs_dataset.py --num-rows 100 --fetch-counts-path ./fetch_counts.json
    python3 prepare_clrs_dataset.py --num-rows 100 # default fetch counts is 100 // 8 = 12 examples per problem type
"""
    print(usage)


def parse_args(reqs: tuple[str, ...] = ("num_rows",)) -> dict[str, int | Path | dict[ProblemType, int]]:
    args = sys.argv[1:]

    config = {}

    def finished_parsing() -> bool:
        return all(key in config for key in reqs)

    def apply_defaults(config: dict[str, int | Path | dict[ProblemType, int]]) -> dict[str, int | Path | dict[ProblemType, int]]:
        defaults = {
            "fetch_counts": {
                pt: config["num_rows"] // len(PROBLEM_TYPES)
                for pt in PROBLEM_TYPES
            },
        }
        return {**defaults, **config}
    
    for i in range(0, len(args), 2):
        if args[i] in ("--num-rows", "-n"):
            config["num_rows"] = int(args[i+1])
        elif args[i] in ("--fetch-counts-path", "-f"):
            fetch_counts_path = Path(args[i+1]).resolve().with_suffix('.json')
            try: 
                with open(fetch_counts_path, 'r') as f:
                    config["fetch_counts"] = json.load(f)
            except: 
                print(f"Error: Invalid fetch counts file: {args[i+1]}")
                sys.exit(1)
        else:
            print(f"Error: Invalid argument: {args[i]}")
            sys.exit(1)

        if finished_parsing():
            break

    if not finished_parsing():
        print(f"Error: Missing required arguments: {', '.join(set(reqs) - set(config.keys()))}")
        print_usage()
        sys.exit(1)

    return apply_defaults(config)


def print_config(config: dict[str, int | Path | dict[ProblemType, int]], indent: str = " " * 4) -> None:
    print("{")
    for i, (k, v) in enumerate(config.items()):
        if k == "fetch_counts":
            print(f"{indent}'fetch_counts':")
            for pt, count in v.items():
                print(f"{indent}{indent}'{pt}': {count},")
        else:
            print(f"{indent}'{k}': {v},")
        if i != len(config) - 1:
            print()
    print("}")


def main() -> None:
    config = parse_args()

    print("Running script with the following configuration:")
    print_config(config)

    prepared_df = prepare_clrs_dataset(**config)

    # print dataset head and save path if being run as an individual script
    print("\nFirst 5 rows of the processed dataset:")
    print(prepared_df.head())
    print(f"\nSaved the processed dataset as a parquet file to path: {SAVE_DIR / 'processed_clrs_dataset.parquet'}")


if __name__ == "__main__":
    main()

import os
import pandas as pd
from pathlib import Path
from problem_mappings import ProblemType as PT, PROBLEM_TYPES
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def parse_df(dataset_file: str|Path) -> pd.DataFrame: 
    return pd.read_parquet(dataset_file)

def count(df: pd.DataFrame, problem_type: PT) -> int:
    filtered = df[df['algo_name'] == problem_type]
    return len(filtered)

def get_problem_types(exclude=None) -> list[PT]:
    return [
        pt for pt in PROBLEM_TYPES
        if exclude is None or pt not in exclude
    ]

def display_counts(parquet_file: str|Path, problem_types: list[PT]) -> None:
    data_df = parse_df(parquet_file)

    freqs = pd.Series(
        [count(data_df, pt) for pt in problem_types],
        index=problem_types,
        name="Frequency"
    )
    percentages = pd.Series(data=freqs * 100 / freqs.sum(), name="Percentage")
    display_df = pd.concat([freqs, percentages], axis=1)

    # round percentages to two decimal places
    display_df["Percentage"] = display_df["Percentage"].map(lambda per: "{per:.2f}%")

    print(display_df)
    print(f"\nTotal count: {freqs.sum()}")


if __name__ == "__main__":
    dataset_file = Path(__file__).parent / "datasets" / "processed_train.parquet"
    display_counts(dataset_file, get_problem_types())

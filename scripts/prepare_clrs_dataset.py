from dataclasses import dataclass
from huggingface_hub import snapshot_download
import os
import pandas as pd
from pathlib import Path
import random
import sys
from types import MappingProxyType

CLRS_TEXT_TRAIN_REPO = 'tomg-group-umd/CLRS-Text-train'
CLRS_TEXT_TEST_REPO = 'tomg-group-umd/CLRS-Text-test'

# use immutable dataclass for mapped problem type space
# to avoid spelling bugs below
@dataclass
class ProblemType: # readonly problem types
    __slots__ = ()
    DIVIDE_CONQUER = "divide_and_conquer"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GEOMETRY = "geometry"
    GRAPHS = "graphs"
    GREEDY = "greedy"
    SEARCH = "search"
    SORTING = "sorting"
    STRINGS = "strings"

PROBLEM_MAPPING = MappingProxyType(dict([
    ("activity_selector", ProblemType.GREEDY),
    ("articulation_points", ProblemType.GRAPHS),
    ("bellman_ford", ProblemType.GRAPHS),
    ("bfs", ProblemType.GRAPHS),
    ("binary_search", ProblemType.SEARCH),
    ("bridges", ProblemType.GRAPHS),
    ("bubble_sort", ProblemType.SORTING),
    ("dag_shortest_paths", ProblemType.GRAPHS),
    ("dfs", ProblemType.GRAPHS),
    ("dijkstra", ProblemType.GRAPHS),
    ("find_maximum_subarray_kadane", ProblemType.DIVIDE_CONQUER),
    ("floyd_warshall", ProblemType.GRAPHS),
    ("graham_scan", ProblemType.GEOMETRY),
    ("heapsort", ProblemType.SORTING),
    ("insertion_sort", ProblemType.SORTING),
    ("jarvis_march", ProblemType.GEOMETRY),
    ("kmp_matcher", ProblemType.STRINGS),
    ("lcs_length", ProblemType.DYNAMIC_PROGRAMMING),
    ("matrix_chain_order", ProblemType.DYNAMIC_PROGRAMMING),
    ("minimum", ProblemType.SEARCH),
    ("mst_kruskal", ProblemType.GRAPHS),
    ("mst_prim", ProblemType.GRAPHS),
    ("naive_string_matcher", ProblemType.STRINGS),
    ("optimal_bst", ProblemType.DYNAMIC_PROGRAMMING),
    ("quickselect", ProblemType.SEARCH),
    ("quicksort", ProblemType.SORTING),
    ("segments_intersect", ProblemType.GEOMETRY),
    ("strongly_connected_components", ProblemType.GRAPHS),
    ("task_scheduling", ProblemType.GREEDY),
    ("topological_sort", ProblemType.GRAPHS),
]))


def download_hf_dataset(hf_repo_id: str) -> str:
    """
    Download parquet files from the given repo to local 
    disk. Return directory (str) in which the downloaded
    files are placed.
    """
    return snapshot_download(repo_id=hf_repo_id, repo_type='dataset', allow_patterns='*.parquet')


def get_parquet_filepaths(root: str) -> list[str]:
    """
    Given root path to huggingface dataset repo,
    extract the .parquet files from the 'data' subdirectory
    and output a list of Path objects corresponding to
    paths to each of the parquet files on local disk.
    """
    data_path = Path(root) / 'data'
    contents = os.listdir(data_path)
    return [
        str(data_path / entry)
        for entry in contents
        if entry.lower().endswith('.parquet')
    ]


def parse_files_to_df(files: list[str], max_rows: int) -> pd.DataFrame:
    """
    Given list of paths to parquet files, parse files
    to dataframes and output the concatenated dataframe
    (must be homogenously-typed)
    """
    assert all(f.lower().endswith('.parquet') for f in files), "Must be a .parquet file!"
    frames = [pd.read_parquet(f) for f in files]
    full_frame = pd.concat(frames, ignore_index=True, sort=False)

    # fetch up to max_rows random rows from full_frame
    random_indices = random.sample(range(len(full_frame)), k=min(max_rows, len(full_frame)))
    truncated_frame = full_frame.loc[random_indices]
    # reset indices so that they are consecutive
    truncated_frame.reset_index(inplace=True, drop=True) 

    return truncated_frame


def map_problem_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given df with 'algo_name' column, map input problem type space
    to corresponding mapped problem type based on the PROBLEM_MAPPING
    dict. Return new df with mapped 'algo_name' column WITHOUT modifying
    input df.
    """
    new_df = df.copy()
    new_df['algo_name'] = df['algo_name'].map(PROBLEM_MAPPING)
    return new_df


def prepare_clrs_dataset(hf_repo_id: str = CLRS_TEXT_TRAIN_REPO, max_rows: int = 10_000):
    """
    Fetch and process CLRS-Text dataset by mapping 30 original
    problem types ('algo_name' column) according to PROBLEM_MAPPING
    and keep up to `max_rows` rows. Return pandas dataframe for
    resulting processed dataset.
    """
    return map_problem_types(
        parse_files_to_df(
            get_parquet_filepaths(download_hf_dataset(hf_repo_id)),
            max_rows
        )
    )


if __name__ == "__main__":
    ### MODIFY THE FOLLOWING LINE TO SWITCH TO TEST REPO ###
    hf_repo_id = CLRS_TEXT_TRAIN_REPO

    using_train = hf_repo_id == CLRS_TEXT_TRAIN_REPO

    save_path = (Path(__file__).parent
                    / "datasets"
                    / f"processed_{"train" if using_train else "test"}.parquet")
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nPreprocessing the {hf_repo_id} dataset...\n")
    df = prepare_clrs_dataset(hf_repo_id=hf_repo_id)
    print(f"\nPreprocessing complete!\n")

    print(df.head())

    df.to_parquet(save_path) # save processed dataset as parquet
    print(f"\nSaved the processed dataset as a parquet file to path: {save_path}")

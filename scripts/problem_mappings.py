from types import MappingProxyType
from dataclasses import dataclass

# use immutable dataclass for mapped problem
# type space to avoid spelling bugs below
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

PROBLEM_TYPES = tuple([
    ProblemType.DIVIDE_CONQUER,
    ProblemType.DYNAMIC_PROGRAMMING,
    ProblemType.GEOMETRY,
    ProblemType.GRAPHS,
    ProblemType.GREEDY,
    ProblemType.SEARCH,
    ProblemType.SORTING,
    ProblemType.STRINGS,
])

# use MappingProxyType to create immutable mapping
# from input problem space to desired problem space
# specified by ProblemType dataclass
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

if __name__ == "__main__":
    pass    

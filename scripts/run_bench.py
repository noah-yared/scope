from pathlib import Path
import sys
import json
from openai import OpenAI

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.problem_mappings import ProblemType

BENCH_DIR = project_root / "benchmark_datasets"
MODEL_OUTPUTS_DIR = project_root / "model_outputs"

def run_benchmark(client: OpenAI, model: str, method: str, size: int):
    with open(BENCH_DIR / f"benchmark_{method}_{size}.json", "r") as f:
        benchmark_dataset = json.load(f)

    results = []
    for idx, item in enumerate(benchmark_dataset):
        print(f"Prompt {idx+1}:")
        messages = [
            {"role": "system", "content": "You are a helpful math assistant."},
            {"role": "user", "content": item["prompt"]}
        ]
        try:
            # Call the v1/chat/completions endpoint
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=8000,
                tools=[],
                tool_choice="none",
            )
            # Extract model output
            model_output = response.choices[0].message.content
        except Exception as e:
            print(f"Error running prompt {idx+1}: {e}")
            model_output = "<answer> response failed </answer>"
        
        print(f"{model_output}\n")

        results.append({
            "algorithm": item["algorithm"],
            "category": item["category"],
            "question": item["question"],
            "answer": item["answer"],
            "model_output": model_output
        })

    with open(MODEL_OUTPUTS_DIR / f"{model}_{method}_{size}.json", "w") as f:
        json.dump(results, f, indent=2)


def main(model: str, size: int, base_url: str, api_key: str):
    client = OpenAI(base_url=base_url, api_key=api_key)
    print(f"Running benchmark for {model} with {size} prompts...\n")
    for method in ['base', 'cot', 'react', 'scope']:
        print(f"Running {method} method...")
        run_benchmark(client=client, size=size, model=model, method=method)
    print(f"\nBenchmark completed successfully!")


if __name__ == "__main__":
    main(model="tei", size=100, base_url="http://0.0.0.0:20000/v1", api_key="sk")

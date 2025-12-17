from pathlib import Path
import re
import json

project_root = Path(__file__).parent.parent

OUTPUTS_DIR = project_root / "model_outputs"

ANSWER_TAG_PATTERN = re.compile(r"<answer>(?<ans>.*)</answer>")

def exact_match(model_answer: str, correct_answer: str) -> bool:
    def strip_answer_tags(text: str) -> str:
        return ANSWER_TAG_PATTERN.search(text).group("ans")
    return strip_answer_tags(model_answer).strip() == correct_answer.strip()


def evaluate_bench(model: str, method: str, size: int) -> float:
    with open(OUTPUTS_DIR / f"{model}_{method}_{size}.json", "r") as f:
        outputs = json.load(f)

    num_correct = sum(
        1 if exact_match(output["model_output"], output["answer"]) else 0
        for output in outputs
    )

    return float(num_correct) / len(outputs)


def main(model: str, size: int) -> None:
    print(f"Evaluating {model} with {size} prompts...")
    for method in ['base', 'cot', 'react', 'scope']:
        accuracy = evaluate_bench(model, method, size)
        print(f"{method} accuracy: {accuracy*100:.2f}% ({round(accuracy*size)}/{size} correct)")
    print(f"Evaluation completed successfully!")


if __name__ == "__main__":
    main(model="tei", size=100)

import json
import math
import random
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATION_DIR = REPO_ROOT / "rag_engine" / "evaluation"

GENERATION_RESULTS = EVALUATION_DIR / "generation_results.json"
RETRIEVAL_RESULTS = EVALUATION_DIR / "retrieval_results.json"
OUTPUT_FILE = EVALUATION_DIR / "bootstrap_results.json"

BOOTSTRAP_SAMPLES = 10_000
CONFIDENCE_LEVEL = 0.95
RANDOM_SEED = 42


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        return json.load(file)


def percentile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower_index = math.floor(position)
    upper_index = math.ceil(position)

    if lower_index == upper_index:
        return values[lower_index]

    weight = position - lower_index
    return (
        values[lower_index] * (1 - weight)
        + values[upper_index] * weight
    )


def bootstrap_mean_ci(
    values: list[float],
    seed: int,
) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot bootstrap an empty list.")

    random_generator = random.Random(seed)
    sample_size = len(values)
    bootstrap_means = []

    for _ in range(BOOTSTRAP_SAMPLES):
        sample_mean = sum(
            values[random_generator.randrange(sample_size)]
            for _ in range(sample_size)
        ) / sample_size

        bootstrap_means.append(sample_mean)

    bootstrap_means.sort()

    alpha = 1 - CONFIDENCE_LEVEL
    lower_bound = percentile(bootstrap_means, alpha / 2)
    upper_bound = percentile(bootstrap_means, 1 - alpha / 2)

    return {
        "sample_size": sample_size,
        "point_estimate": round(sum(values) / sample_size, 4),
        "ci_95_lower": round(lower_bound, 4),
        "ci_95_upper": round(upper_bound, 4),
    }


def main() -> None:
    generation_results = load_json(GENERATION_RESULTS)
    retrieval_results = load_json(RETRIEVAL_RESULTS)

    generation_questions = generation_results["questions"]
    retrieval_questions = retrieval_results["questions"]

    answer_f1_values = [
        float(question["answer_f1"])
        for question in generation_questions
        if question["is_answerable"]
        and question["answer_f1"] is not None
    ]

    hallucination_values = [
        1.0 if question["outcome"] == "hallucination" else 0.0
        for question in generation_questions
        if not question["is_answerable"]
        and question["outcome"] != "generation_error"
    ]

    top_k = int(retrieval_results["configuration"]["top_k"])

    evidence_recall_values = [
        1.0
        if question["evidence_hit_rank"] is not None
        and question["evidence_hit_rank"] <= top_k
        else 0.0
        for question in retrieval_questions
    ]

    result = {
        "method": {
            "name": "nonparametric percentile bootstrap",
            "confidence_level": CONFIDENCE_LEVEL,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "random_seed": RANDOM_SEED,
        },
        "metrics": {
            "answer_f1": bootstrap_mean_ci(
                answer_f1_values,
                seed=RANDOM_SEED,
            ),
            "hallucination_rate": bootstrap_mean_ci(
                hallucination_values,
                seed=RANDOM_SEED + 1,
            ),
            f"evidence_recall_at_{top_k}": bootstrap_mean_ci(
                evidence_recall_values,
                seed=RANDOM_SEED + 2,
            ),
        },
    }

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\nResults saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
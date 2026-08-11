import argparse
import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import Any

from rag_engine import agent
from rag_engine.agent.rag_agent import RAGAgent
from rag_engine.evaluation.evaluate_retrieval import (
    load_json,
    prepare_evaluation_store,
    repository_relative_path,
)


UNANSWERABLE = "UNANSWERABLE"
GENERATION_ERROR = "GENERATION_ERROR"

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "rag_engine" / "data"
EVALUATION_DIR = ROOT_DIR / "rag_engine" / "evaluation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the generation part of the RAG pipeline."
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DATA_DIR / "sample_questions.json",
    )
    parser.add_argument(
        "--chunks",
        type=Path,
        default=DATA_DIR / "sample_chunks.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Output JSON path. If omitted, the filename is selected "
            "from the prompt variant."
        ),
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("baseline", "improved"),
        default="baseline",
    )
    parser.add_argument(
        "--chroma-dir",
        type=Path,
        default=ROOT_DIR / "chroma_db",
    )
    parser.add_argument(
        "--collection",
        default="rag_generation_evaluation",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5:3b",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
    )
    return parser.parse_args()


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(character for character in text if character not in string.punctuation)
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def token_f1(prediction: str, reference: str) -> float:
    prediction_tokens = normalize_answer(prediction).split()
    reference_tokens = normalize_answer(reference).split()

    if not prediction_tokens and not reference_tokens:
        return 1.0

    if not prediction_tokens or not reference_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(reference_tokens)
    common_count = sum(common.values())

    if common_count == 0:
        return 0.0

    precision = common_count / len(prediction_tokens)
    recall = common_count / len(reference_tokens)

    return 2 * precision * recall / (precision + recall)


def max_answer_f1(prediction: str, references: list[str]) -> float:
    if not references:
        return 0.0

    return max(token_f1(prediction, reference) for reference in references)


def evaluate(
    questions: list[dict[str, Any]],
    agent: RAGAgent,
    top_k: int,
    model: str,
    prompt_variant: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = []
    answer_f1_scores = []

    completed_unanswerable = 0
    hallucination_count = 0
    generation_errors = 0

    for number, question in enumerate(questions, start=1):
        retrieved_chunks = agent.ask_question(
            question["question"],
            top_k=top_k,
            show_table=False,
        )

        documents = [chunk["text"] for chunk in retrieved_chunks]

        prediction = agent.generate_answer(
            question["question"],
            documents,
            model=model,
            prompt_variant=prompt_variant,
        )

        is_answerable = bool(question.get("is_answerable"))
        answer_f1 = None

        if prediction == GENERATION_ERROR:
            outcome = "generation_error"
            generation_errors += 1

        elif is_answerable:
            answer_f1 = max_answer_f1(
                prediction,
                question["answers"],
            )
            answer_f1_scores.append(answer_f1)

            if prediction == UNANSWERABLE:
                outcome = "false_abstention"
            else:
                outcome = "answered"

        else:
            completed_unanswerable += 1

            if prediction == UNANSWERABLE:
                outcome = "correct_abstention"
            else:
                outcome = "hallucination"
                hallucination_count += 1

        records.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "is_answerable": is_answerable,
                "reference_answers": question["answers"],
                "prediction": prediction,
                "answer_f1": answer_f1,
                "outcome": outcome,
                "retrieved_chunks": retrieved_chunks,
            }
        )

        print(
            f"Evaluated {number}/{len(questions)} questions",
            end="\r",
            flush=True,
        )

    print()

    answerable_count = sum(
        1 for question in questions if question.get("is_answerable")
    )
    unanswerable_count = len(questions) - answerable_count

    average_answer_f1 = (
        sum(answer_f1_scores) / len(answer_f1_scores)
        if answer_f1_scores
        else None
    )

    hallucination_rate = (
        hallucination_count / completed_unanswerable
        if completed_unanswerable
        else None
    )

    summary = {
        "evaluated_questions": len(questions),
        "answerable_questions": answerable_count,
        "unanswerable_questions": unanswerable_count,
        "completed_answerable_questions": len(answer_f1_scores),
        "completed_unanswerable_questions": completed_unanswerable,
        "generation_errors": generation_errors,
        "answer_f1": (
            round(average_answer_f1, 4)
            if average_answer_f1 is not None
            else None
        ),
        "hallucination_count": hallucination_count,
        "hallucination_rate": (
            round(hallucination_rate, 4)
            if hallucination_rate is not None
            else None
        ),
    }

    return summary, records


def main() -> None:
    args = parse_args()

    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    questions = load_json(args.questions)
    chunks = load_json(args.chunks)

    embedder, database = prepare_evaluation_store(
        chunks,
        args.collection,
        args.chroma_dir,
    )

    agent = RAGAgent(
        embedder=embedder,
        db=database,
    )

    output_path = args.output
    if output_path is None:
        if args.prompt_variant == "baseline":
            output_path = EVALUATION_DIR / "generation_results.json"
        else:
            output_path = (
                EVALUATION_DIR
                / "generation_results_improved_prompt.json"
            )

    summary, records = evaluate(
        questions,
        agent,
        args.top_k,
        args.model,
        args.prompt_variant,
    )

    result = {
        "configuration": {
            "questions_file": repository_relative_path(args.questions),
            "chunks_file": repository_relative_path(args.chunks),
            "model": args.model,
            "prompt_variant": args.prompt_variant,
            "collection": args.collection,
            "top_k": args.top_k,
        },
        "metric_definitions": {
            "answer_f1": (
                "Token-level F1 between the generated answer and the best "
                "reference answer for answerable questions."
            ),
            "hallucination_rate": (
                "The proportion of completed unanswerable questions for which "
                "the model generated an answer instead of UNANSWERABLE."
            ),
        },
        "summary": summary,
        "questions": records,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main()
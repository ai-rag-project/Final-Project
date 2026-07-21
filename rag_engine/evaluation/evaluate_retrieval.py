"""Run this file from the repository root:
    python rag_engine/evaluation/evaluate_retrieval.py
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import sys
import unicodedata
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_QUESTIONS = REPO_ROOT / "rag_engine" / "data" / "sample_questions.json"
DEFAULT_CHUNKS = REPO_ROOT / "rag_engine" / "data" / "sample_chunks.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "retrieval_results.json"
DEFAULT_CHROMA_DIR = REPO_ROOT / "chroma_db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure retrieval recall on answerable SQuAD questions."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--collection", default="rag_evaluation")
    parser.add_argument("--chroma-dir", type=Path, default=DEFAULT_CHROMA_DIR)
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Required data file does not exist: {path}")

    with path.open(encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return data


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    return re.sub(r"\s+", " ", text).strip()


def first_context_hit_rank(
    metadatas: list[dict[str, Any]], gold_context_id: int
) -> int | None:
    for rank, metadata in enumerate(metadatas, start=1):
        if metadata.get("context_id") == gold_context_id:
            return rank
    return None


def first_evidence_hit_rank(documents: list[str], answers: list[str]) -> int | None:
    normalized_answers = [normalize_text(answer) for answer in answers]
    normalized_answers = [answer for answer in normalized_answers if answer]

    for rank, document in enumerate(documents, start=1):
        normalized_document = normalize_text(document)
        if any(answer in normalized_document for answer in normalized_answers):
            return rank
    return None


def recall_at(records: list[dict[str, Any]], rank_field: str, k: int) -> float:
    if not records:
        return 0.0
    hits = sum(
        record[rank_field] is not None and record[rank_field] <= k
        for record in records
    )
    return hits / len(records)


def prepare_evaluation_store(
    chunks: list[dict[str, Any]], collection_name: str, chroma_dir: Path
):
    from rag_engine.embedding.embedder import LocalEmbedder
    from rag_engine.vector_store.repository import VectorRepository

    required_fields = {"id", "context_id", "text"}
    for index, chunk in enumerate(chunks):
        missing = required_fields - chunk.keys()
        if missing:
            raise ValueError(f"Chunk {index} is missing fields: {sorted(missing)}")

    embedder = LocalEmbedder()
    database = VectorRepository(
        collection_name=collection_name,
        persist_directory=str(chroma_dir),
    )

    database.clear_collection()

    ids = [str(chunk["id"]) for chunk in chunks]
    documents = [chunk["text"] for chunk in chunks]
    metadatas = [{"context_id": chunk["context_id"]} for chunk in chunks]
    embeddings = embedder.embed_documents(documents)
    database.add_chunks(ids, embeddings, documents, metadatas)

    return embedder, database


def evaluate(
    questions: list[dict[str, Any]],
    embedder,
    database,
    top_k: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    answerable = [question for question in questions if question.get("is_answerable")]
    if not answerable:
        raise ValueError("No answerable questions were found.")

    records: list[dict[str, Any]] = []

    for number, question in enumerate(answerable, start=1):
        query_embedding = embedder.embed_query(question["question"])
        search_result = database.search_similar(query_embedding, n_results=top_k)

        documents = search_result.get("documents", [[]])[0]
        metadatas = search_result.get("metadatas", [[]])[0]
        ids = search_result.get("ids", [[]])[0]
        distances = search_result.get("distances", [[]])[0]

        context_rank = first_context_hit_rank(metadatas, question["context_id"])
        evidence_rank = first_evidence_hit_rank(documents, question["answers"])

        retrieved_chunks = []
        for index, document in enumerate(documents):
            retrieved_chunks.append(
                {
                    "rank": index + 1,
                    "id": ids[index] if index < len(ids) else None,
                    "context_id": (
                        metadatas[index].get("context_id")
                        if index < len(metadatas)
                        else None
                    ),
                    "distance": distances[index] if index < len(distances) else None,
                    "text": document,
                }
            )

        records.append(
            {
                "question_id": question["id"],
                "question": question["question"],
                "question_word_count": len(question["question"].split()),
                "gold_context_id": question["context_id"],
                "reference_answers": question["answers"],
                "context_hit_rank": context_rank,
                "evidence_hit_rank": evidence_rank,
                "retrieved_chunks": retrieved_chunks,
            }
        )
        print(f"Evaluated {number}/{len(answerable)} questions", end="\r", flush=True)

    print()

    median_length = statistics.median(
        record["question_word_count"] for record in records
    )
    short_records = [
        record for record in records if record["question_word_count"] <= median_length
    ]
    long_records = [
        record for record in records if record["question_word_count"] > median_length
    ]

    summary = {
        "evaluated_questions": len(records),
        "top_k": top_k,
        "context_recall_at_1": recall_at(records, "context_hit_rank", 1),
        f"context_recall_at_{top_k}": recall_at(records, "context_hit_rank", top_k),
        "evidence_recall_at_1": recall_at(records, "evidence_hit_rank", 1),
        f"evidence_recall_at_{top_k}": recall_at(records, "evidence_hit_rank", top_k),
        "question_length_breakdown": {
            "median_word_count": median_length,
            "short": {
                "definition": f"word_count <= {median_length}",
                "count": len(short_records),
                f"evidence_recall_at_{top_k}": recall_at(
                    short_records, "evidence_hit_rank", top_k
                ),
            },
            "long": {
                "definition": f"word_count > {median_length}",
                "count": len(long_records),
                f"evidence_recall_at_{top_k}": recall_at(
                    long_records, "evidence_hit_rank", top_k
                ),
            },
        },
    }
    return summary, records


def main() -> None:
    args = parse_args()
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")

    questions = load_json(args.questions)
    chunks = load_json(args.chunks)
    embedder, database = prepare_evaluation_store(
        chunks, args.collection, args.chroma_dir
    )
    summary, records = evaluate(questions, embedder, database, args.top_k)

    result = {
        "configuration": {
            "questions_file": str(args.questions),
            "chunks_file": str(args.chunks),
            "collection": args.collection,
            "top_k": args.top_k,
            "answerable_only": True,
        },
        "metric_definitions": {
            "context_recall": (
                "A hit occurs when a retrieved chunk has the gold context_id."
            ),
            "evidence_recall": (
                "A hit occurs when a normalized reference answer appears in a "
                "retrieved chunk. This is the primary retrieval metric."
            ),
        },
        "summary": summary,
        "questions": records,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()
import json
import random
import urllib.request
from pathlib import Path


DATA_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
DATA_DIR = Path(__file__).resolve().parent
RAW_FILE = DATA_DIR / "dev-v2.0.json"

TARGET_WORDS = 7000
CHUNK_SIZE = 70
OVERLAP = 15
STEP = CHUNK_SIZE - OVERLAP


if not RAW_FILE.exists():
    print("Downloading SQuAD 2.0...")
    urllib.request.urlretrieve(DATA_URL, RAW_FILE)

with RAW_FILE.open(encoding="utf-8") as file:
    squad = json.load(file)


paragraphs = []
for article in squad["data"]:
    for paragraph in article["paragraphs"]:
        paragraphs.append(
            {
                "title": article["title"],
                "context": paragraph["context"],
                "qas": paragraph["qas"],
            }
        )


random.seed(42)
random.shuffle(paragraphs)


selected_contexts = []
total_words = 0

for paragraph in paragraphs:
    context_words = paragraph["context"].split()
    selected_contexts.append(paragraph)
    total_words += len(context_words)

    if total_words >= TARGET_WORDS:
        break


chunks = []

for context_id, paragraph in enumerate(selected_contexts):
    words = paragraph["context"].split()

    for start in range(0, len(words), STEP):
        chunk_words = words[start : start + CHUNK_SIZE]

        if not chunk_words:
            break

        chunks.append(
            {
                "id": len(chunks),
                "context_id": context_id,
                "text": " ".join(chunk_words),
            }
        )

        if start + CHUNK_SIZE >= len(words):
            break


answerable_questions = []
unanswerable_questions = []

for context_id, paragraph in enumerate(selected_contexts):
    for qa in paragraph["qas"]:
        question = {
            "id": qa["id"],
            "context_id": context_id,
            "question": qa["question"],
            "answers": [answer["text"] for answer in qa["answers"]],
            "is_answerable": not qa["is_impossible"],
        }

        if question["is_answerable"]:
            answerable_questions.append(question)
        else:
            unanswerable_questions.append(question)


random.shuffle(answerable_questions)
random.shuffle(unanswerable_questions)
questions = answerable_questions[:50] + unanswerable_questions[:50]
random.shuffle(questions)


contexts_for_output = []
for context_id, paragraph in enumerate(selected_contexts):
    contexts_for_output.append(
        {
            "id": context_id,
            "title": paragraph["title"],
            "text": paragraph["context"],
        }
    )


with open(DATA_DIR / "sample_contexts.json", "w", encoding="utf-8") as file:
    json.dump(contexts_for_output, file, ensure_ascii=False, indent=2)

with open(DATA_DIR / "sample_chunks.json", "w", encoding="utf-8") as file:
    json.dump(chunks, file, ensure_ascii=False, indent=2)

with open(DATA_DIR / "sample_questions.json", "w", encoding="utf-8") as file:
    json.dump(questions, file, ensure_ascii=False, indent=2)


print("Contexts:", len(contexts_for_output))
print("Words:", total_words)
print("Chunks:", len(chunks))
print("Questions:", len(questions))

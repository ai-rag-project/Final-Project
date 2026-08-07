# YadYar Lite — RAG and Hallucination Study

A lightweight course project for **T-05: Retrieval-Augmented Generation and the Study of Hallucination**.

This project studies a RAG pipeline by evaluating its retrieval and generation components separately. The current version implements a reproducible dense-retrieval baseline; generator integration and end-to-end hallucination evaluation are still in progress.

## Project Question

How well can a lightweight RAG system retrieve answer evidence from a small document collection and avoid unsupported answers when sufficient evidence is unavailable?

## Dataset

The project uses a reproducible subset of the **SQuAD 2.0 development set**.

| Item | Value |
|---|---:|
| Source contexts | 56 |
| Context words | 7,144 |
| Retrieval chunks | 146 |
| Chunk size | 70 words |
| Chunk overlap | 15 words |
| Answerable questions | 50 |
| Unanswerable questions | 50 |

The preprocessing script uses random seed `42`, making the generated subset reproducible.

## Retrieval Baseline

The retriever uses:

- `BAAI/bge-base-en-v1.5` for dense embeddings
- ChromaDB as the vector store
- Cosine distance for similarity search
- `Top-k = 3` retrieved chunks per question

The current pipeline is:

```text
Question
   -> Query embedding
   -> ChromaDB similarity search
   -> Top-3 retrieved chunks
   -> Generator (in progress)
   -> Answer or UNANSWERABLE
```

No model is trained from scratch and no fine-tuning is used.

## Current Retrieval Results

The baseline was evaluated on 50 answerable questions.

| Metric | Result |
|---|---:|
| Context Recall@1 | 0.98 |
| Context Recall@3 | 0.98 |
| Evidence Recall@1 | 0.86 |
| Evidence Recall@3 | **0.98** |

The retriever found answer evidence in the Top-3 chunks for **49 of 50 questions**.

### Question-Length Breakdown

| Question group | Count | Evidence Recall@3 |
|---|---:|---:|
| Short questions (`<= 9` words) | 29 | 0.966 |
| Long questions (`> 9` words) | 21 | 1.000 |

Detailed results are stored in:

```text
rag_engine/evaluation/retrieval_results.json
```

## Project Structure

```text
Final-Project/
├── docs/
│   └── phase1_report.md
├── rag_engine/
│   ├── agent/
│   │   └── rag_agent.py
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── sample_contexts.json
│   │   ├── sample_chunks.json
│   │   └── sample_questions.json
│   ├── embedding/
│   │   └── embedder.py
│   ├── evaluation/
│   │   ├── evaluate_retrieval.py
│   │   └── retrieval_results.json
│   └── vector_store/
│       └── repository.py
├── main.py
├── visualize.py
└── requirements.txt
```

## Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/ai-rag-project/Final-Project.git
cd Final-Project
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

The embedding model is downloaded automatically from Hugging Face on the first run. No API key is required.

## Data Preparation

The prepared dataset files are included in the repository. To reproduce them from the original SQuAD 2.0 development set, run:

```bash
python rag_engine/data/prepare_data.py
```

This command downloads the original dataset when necessary and regenerates the contexts, chunks, and evaluation questions.

## Retrieval Evaluation

Run the retrieval evaluator from the repository root:

```bash
python rag_engine/evaluation/evaluate_retrieval.py
```

The evaluator:

1. Loads the prepared chunks and questions.
2. Rebuilds a separate ChromaDB evaluation collection.
3. Evaluates the 50 answerable questions.
4. Prints the metric summary.
5. Saves detailed results to `retrieval_results.json`.

## Interactive Retrieval Demo

Run:

```bash
python main.py
```

The script:

- embeds and stores the prepared chunks;
- displays an interactive t-SNE visualization;
- accepts questions through the terminal;
- prints the three most relevant retrieved chunks.

At the current stage, the interactive demo displays retrieved evidence rather than a generated final answer.

## Evaluation Plan

The completed retrieval stage uses **Evidence Recall@3** as its main metric.

After generator integration, the project will additionally measure:

- token-level Answer F1;
- correct abstention on unanswerable questions;
- hallucination rate;
- answerable versus unanswerable performance.

Representative failures will be grouped into:

1. Retrieval failure
2. Ignored-evidence hallucination
3. Absent-evidence hallucination

## Current Limitations

- The dataset and evaluation sample are intentionally small.
- Fixed word-based chunking can split relevant sentences.
- Evidence matching relies on normalized reference-answer occurrence.
- The current interactive pipeline does not yet generate final answers.
- End-to-end hallucination and abstention results require generator integration.

## Report

The Phase 1 report is available at:

[`docs/phase1_report.md`](docs/phase1_report.md)

## Team

- Taha Amini
- Eiliya Yavari

Artificial Intelligence and Expert Systems — Spring 1404–1405
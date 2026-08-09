# YadYar Lite - RAG and Hallucination Study

A lightweight course project for **T-05: Retrieval-Augmented Generation and the Study of Hallucination**.

This project evaluates a local retrieval-augmented generation (RAG) pipeline in two stages: whether the retriever finds answer evidence and whether the generator produces a supported answer or correctly returns `UNANSWERABLE` when the evidence is insufficient.

## Table of Contents

- [Project Question](#project-question)
- [Dataset](#dataset)
- [Baseline System](#baseline-system)
- [Baseline Results](#baseline-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Evaluation](#evaluation)
- [End-to-End Demo](#end-to-end-demo)
- [Representative Errors](#representative-errors)
- [Current Limitations](#current-limitations)
- [Reports](#reports)
- [Team](#team)

## Project Question

How well can a lightweight RAG system find answer evidence in a small document collection, generate answers supported by that evidence, and avoid answering when sufficient evidence is unavailable?

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
| Random seed | 42 |

The prepared questions retain their SQuAD IDs, source context IDs, reference answers, and answerability labels.

## Baseline System

The local baseline uses:

- `BAAI/bge-base-en-v1.5` for dense embeddings;
- ChromaDB with cosine distance as the vector store;
- `Top-k = 3` retrieved chunks per question;
- `qwen2.5:3b`, served locally through Ollama, for answer generation.

The complete pipeline is:

```text
Question
   -> BGE query embedding
   -> ChromaDB Top-3 retrieval
   -> Qwen2.5 generation
   -> answer, UNANSWERABLE, or GENERATION_ERROR
```

The generator receives only the question and the three retrieved chunks. Its prompt instructs it to use the supplied evidence, produce a short answer, and return exactly `UNANSWERABLE` when the context is insufficient.

No model is trained from scratch and no fine-tuning is used.

## Baseline Results

### Retrieval

The retriever was evaluated on the 50 answerable questions.

| Metric | Result |
|---|---:|
| Context Recall@1 | 0.98 |
| Context Recall@3 | 0.98 |
| Evidence Recall@1 | 0.86 |
| Evidence Recall@3 | **0.98** |

The retriever found normalized reference-answer evidence in the Top-3 chunks for **49 of 50 answerable questions**.

#### Question-Length Breakdown

The median answerable-question length was nine words.

| Question group | Count | Evidence found | Evidence Recall@3 |
|---|---:|---:|---:|
| Short (`<= 9` words) | 29 | 28 | 0.966 |
| Long (`> 9` words) | 21 | 21 | 1.000 |

### Generation

The generator was evaluated on all 100 questions. All generation requests completed successfully.

| Metric | Result | Interpretation |
|---|---:|---|
| Answer F1 | **0.6825** | Average token overlap on the 50 answerable questions |
| Hallucination Rate | **0.44** | 22 of 50 unanswerable questions received a normal answer |
| Correct Abstention Rate | **0.56** | 28 of 50 unanswerable questions returned `UNANSWERABLE` |
| Generation Errors | **0** | All 100 requests completed |

Among the 50 answerable questions, the generator produced a normal answer for 46 and incorrectly abstained on 4. Retrieval coverage was strong, but generation and abstention remained the main sources of error.

Detailed results are stored in:

```text
rag_engine/evaluation/retrieval_results.json
rag_engine/evaluation/generation_results.json
```

## Project Structure

```text
Final-Project/
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ phase1_report.md
â”‚   â””â”€â”€ phase2_report.md
â”œâ”€â”€ rag_engine/
â”‚   â”œâ”€â”€ agent/
â”‚   â”‚   â””â”€â”€ rag_agent.py
â”‚   â”œâ”€â”€ config/
â”‚   â”‚   â””â”€â”€ config.py
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ prepare_data.py
â”‚   â”‚   â”œâ”€â”€ sample_contexts.json
â”‚   â”‚   â”œâ”€â”€ sample_chunks.json
â”‚   â”‚   â””â”€â”€ sample_questions.json
â”‚   â”œâ”€â”€ embedding/
â”‚   â”‚   â””â”€â”€ embedder.py
â”‚   â”œâ”€â”€ evaluation/
â”‚   â”‚   â”œâ”€â”€ evaluate_retrieval.py
â”‚   â”‚   â”œâ”€â”€ evaluate_generation.py
â”‚   â”‚   â”œâ”€â”€ retrieval_results.json
â”‚   â”‚   â””â”€â”€ generation_results.json
â”‚   â””â”€â”€ vector_store/
â”‚       â””â”€â”€ repository.py
â”œâ”€â”€ main.py
â”œâ”€â”€ visualize.py
â””â”€â”€ requirements.txt
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
.\.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```cmd
.venv\Scripts\activate.bat
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install the Python dependencies:

```bash
python -m pip install -r requirements.txt
```

The embedding model is downloaded automatically from Hugging Face on the first run. No Hugging Face API token is required.

### Local Generator with Ollama

Install Ollama from [ollama.com/download](https://ollama.com/download), then pull the model used by this project:

```bash
ollama pull qwen2.5:3b
```

Make sure the Ollama service is running. If it is not already running in the background, start it with:

```bash
ollama serve
```

No external generation API key is required.

## Data Preparation

The prepared dataset files are included in the repository. To reproduce them from the original SQuAD 2.0 development set, run:

```bash
python rag_engine/data/prepare_data.py
```

This command downloads the original dataset when necessary and regenerates the contexts, chunks, and evaluation questions using random seed `42`.

## Evaluation

Run both evaluators from the repository root:

```bash
python -m rag_engine.evaluation.evaluate_retrieval
python -m rag_engine.evaluation.evaluate_generation
```

The retrieval evaluator rebuilds a separate ChromaDB evaluation collection, processes the 50 answerable questions, prints the retrieval metrics, and saves the per-question results.

The generation evaluator processes all 100 questions, computes Answer F1 and Hallucination Rate, prints the metric summary, and saves the generated answers together with their retrieved chunks.

Generation evaluation requires the local Ollama service and `qwen2.5:3b` model.

## End-to-End Demo

After installing the dependencies and pulling the Ollama model, run:

```bash
python main.py
```

The terminal demo:

1. embeds and stores the prepared chunks;
2. accepts a question from the user;
3. retrieves the three most similar chunks;
4. displays the retrieved evidence;
5. generates a final answer, `UNANSWERABLE`, or `GENERATION_ERROR`.

Example supported question:

```text
When did Ribault first establish a settlement in South Carolina?
```

Expected answer:

```text
1562
```

An unrelated question with no supporting evidence should return `UNANSWERABLE`, although the evaluation shows that abstention is not yet reliable in every case.

## Representative Errors

Manual review identified several recurring failure patterns:

- ambiguous standalone questions causing retrieval failure;
- false abstention even when the correct evidence was retrieved;
- extraction of a nearby phrase instead of the requested value;
- answering questions that contain a false premise;
- incorrect handling of negation;
- token-level F1 mismatches for equivalent forms such as `2` and `two`.

Representative examples and explanations are included in [`docs/phase2_report.md`](docs/phase2_report.md).

## Current Limitations

- The corpus contains only 56 contexts and 100 evaluation questions.
- Only one embedding model, one generator, one prompt, and `Top-k = 3` were evaluated.
- Fixed word-based chunking can split sentences or separate useful context.
- Evidence Recall@3 relies on normalized reference-answer occurrence and may miss paraphrased evidence.
- Token-level Answer F1 does not recognize every semantically equivalent answer.
- Hallucination Rate measures failure to abstain on labeled unanswerable questions; it is not an independent factuality check.
- Some SQuAD questions depend on their original paragraph and are ambiguous as standalone retrieval queries.
- The system has no separate evidence verifier for contradictions, false premises, or unsupported claims.

## Reports

- [Phase 1 report](docs/phase1_report.md)
- [Phase 2 report](docs/phase2_report.md)

## Team

- Taha Amini
- Eiliya Yavari

Artificial Intelligence and Expert Systems - Spring 1404-1405
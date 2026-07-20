# SQuAD 2.0 Sample Data

This folder contains the sample data used in our T-05 RAG project.

## Dataset

We use the development split of SQuAD 2.0. SQuAD is a question-answering
dataset made from Wikipedia paragraphs. It contains both answerable and
unanswerable questions. This makes it suitable for studying hallucination,
because the model should not invent an answer when the context does not contain
enough information.

Official dataset page:
https://rajpurkar.github.io/SQuAD-explorer/

## Data preparation

The `prepare_data.py` script performs these steps:

1. Downloads the official `dev-v2.0.json` file if it is not available.
2. Shuffles the SQuAD paragraphs using random seed 42.
3. Selects paragraphs until the sample contains about 7,000 words.
4. Divides every context into chunks of 70 words.
5. Uses an overlap of 15 words between consecutive chunks.
6. Selects 50 answerable and 50 unanswerable questions.

The step between two chunks is:

```text
70 - 15 = 55 words
```

## Output files

Running the script creates these files:

- `sample_contexts.json`: 56 selected SQuAD contexts containing 7,144 words.
- `sample_chunks.json`: 146 chunks used by the retrieval system.
- `sample_questions.json`: 100 questions for later evaluation.

## Run

From the root of the repository, run:

```powershell
python rag_engine/data/prepare_data.py
```

The downloaded `dev-v2.0.json` file is only raw input data and does not need to
be committed to GitHub.

## AI tool usage

OpenAI ChatGPT was used to help prepare and review the data preprocessing
script. The source data itself comes from the official SQuAD 2.0 dataset.

## License

SQuAD 2.0 is distributed under the CC BY-SA 4.0 license.

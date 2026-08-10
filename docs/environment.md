# Reproducibility Environment

The reported project results were produced using the following environment.

## Python Environment

- Operating system: Windows 11
- Python: 3.13.4
- pip: 26.0.1
- PyTorch: 2.13.0+cpu
- CUDA available: No
- Retrieval execution device: CPU

The complete Python environment snapshot is available in
[`requirements-lock.txt`](../requirements-lock.txt).

To install the exact recorded package versions:

```bash
python -m pip install -r requirements-lock.txt
```

## Retrieval Environment

- Embedding model: `BAAI/bge-base-en-v1.5`
- Hugging Face revision: `a5beb1e3e68b9ab74eb54cfd186867f64f240e1a`
- `sentence-transformers`: 5.6.0
- `chromadb`: 1.5.9
- `torch`: 2.13.0+cpu

## Generation Environment

- Runtime: Ollama
- Ollama client version: 0.32.6
- Ollama server version: 0.32.6
- Model tag: `qwen2.5:3b`
- Model ID reported by Ollama: `357c53fb659c`
- Architecture: Qwen2
- Parameters: 3.1B
- Quantization: `Q4_K_M`
- Context length: 32,768 tokens

## Reproducibility Notes

- The embedding model revision is recorded so that the exact model version
  used for the reported results can be identified.
- `requirements-lock.txt` is an environment snapshot generated with
  `pip freeze` on Windows.
- The original evaluation artifacts are committed to the repository so that
  the reported metrics can be inspected without rerunning the models.
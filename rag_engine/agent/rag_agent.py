import json
import os
import requests
from typing import List, Dict, Any

from rich.console import Console
from rich.table import Table

from rag_engine.embedding.embedder import LocalEmbedder
from rag_engine.vector_store.repository import VectorRepository

console = Console()


class RAGAgent:
    def __init__(self):
        console.print("[dim][*] Initializing RAG Agent...[/dim]")
        self.embedder = LocalEmbedder()
        self.db = VectorRepository()

    def ingest_data(self, json_path: str = None):
        """
        Reads chunked data from rag_engine/data/ file, embeds the text, and stores in the ChromaDB.
        """
        if json_path is None:
            base_dir = os.path.dirname(os.path.dirname(__file__))
            json_path = os.path.join(base_dir, "data", "sample_chunks.json")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"[-] The file {json_path} does not exist.")

        console.print(f"[dim][*] Reading data from {json_path}...[/dim]")

        with open(json_path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)

        ids: List[str] = []
        texts: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks_data:
            # IDs need to be strings
            ids.append(str(chunk["id"]))
            texts.append(chunk["text"])
            metadatas.append({"context_id": chunk["context_id"]})

        console.print(f"[green][+] Successfully loaded {len(texts)} chunks from JSON.[/green]")

        console.print("[dim][*] Generating embeddings for the chunks...[/dim]")
        embeddings = self.embedder.embed_documents(texts)
        console.print("[green][+] Embeddings generated successfully.[/green]")

        console.print("[dim][*] Storing data in ChromaDB...[/dim]")
        self.db.add_chunks(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        console.print("[green][+] Data ingestion complete! DB is ready.[/green]")

    def ask_question(self, question: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the question, retrieves the top_k most similar chunks from ChromaDB,
        and prints a clean summary table (chunk ID + similarity only).
        """
        question_embedding = self.embedder.embed_query(question)

        search_results = self.db.search_similar(
            query_embedding=question_embedding,
            n_results=top_k,
        )

        retrieved_documents = search_results.get("documents", [[]])[0]
        retrieved_metadatas = search_results.get("metadatas", [[]])[0]
        retrieved_ids = search_results.get("ids", [[]])[0]
        retrieved_distances = search_results.get("distances", [[]])[0] if search_results.get("distances") else []

        formatted_chunks = []

        table = Table(title=f"Retrieved Chunks ({len(retrieved_documents)})", show_lines=False, header_style="bold blue")
        table.add_column("#", justify="right", style="dim")
        table.add_column("Chunk ID", style="cyan")
        table.add_column("Context ID", style="magenta")
        table.add_column("Similarity", justify="right")

        for i in range(len(retrieved_documents)):
            distance = retrieved_distances[i] if i < len(retrieved_distances) else None
            similarity_percentage = round((1 - distance) * 100, 2) if distance is not None else None
            metadata = retrieved_metadatas[i] if i < len(retrieved_metadatas) else {}

            chunk_data = {
                "id": retrieved_ids[i] if i < len(retrieved_ids) else None,
                "metadata": metadata,
                "text": retrieved_documents[i],
                "distance": distance,
                "similarity_percentage": similarity_percentage,
            }
            formatted_chunks.append(chunk_data)

            if similarity_percentage is None:
                sim_style = "dim"
                sim_display = "N/A"
            elif similarity_percentage >= 55:
                sim_style = "green"
                sim_display = f"{similarity_percentage}%"
            elif similarity_percentage >= 45:
                sim_style = "yellow"
                sim_display = f"{similarity_percentage}%"
            else:
                sim_style = "red"
                sim_display = f"{similarity_percentage}%"

            table.add_row(
                str(i + 1),
                str(chunk_data["id"]),
                str(metadata.get("context_id", "-")),
                f"[{sim_style}]{sim_display}[/{sim_style}]",
            )

        console.print(table)

        return formatted_chunks

    def generate_answer(self, question: str, chunks: List[str], model: str = "qwen2.5:3b") -> str:

        context = "\n\n".join(chunks)
        prompt = f"""You are a question-answering assistant. Answer the question using ONLY the information in the context below. Do not use any outside knowledge.

        Rules:
        - If the context does NOT contain enough information, respond with exactly: UNANSWERABLE
        - Do not use any outside knowledge.

        Context:
        {context}

        Question: {question}
        Answer:"""

        try:
            with console.status("[dim]Generating answer...[/dim]", spinner="dots"):
                response = requests.post(
                    "http://localhost:11434/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                    timeout=60,
                )
            response.raise_for_status()
            return response.json()["response"].strip()
        except requests.exceptions.RequestException as e:
            console.print(f"[red][-] Generation failed: {e}[/red]")
            return "GENERATION_ERROR"
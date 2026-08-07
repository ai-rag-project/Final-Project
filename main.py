from rich.console import Console
from rich.panel import Panel


from rag_engine.agent.rag_agent import RAGAgent
from visualize import visualize_embeddings

console = Console()


def main():
    console.rule("[bold cyan]RAG Pipeline Startup")

    agent = RAGAgent()

    console.print("\n[dim][*] Beginning data ingestion process...[/dim]")
    agent.ingest_data(None)  # None for default data path

    console.print("[green][+] Setup and Ingestion finished successfully![/green]")

    visualize_embeddings()
    console.print("[green][+] Visualizer launched in your browser.[/green]")

    console.rule("[bold cyan]RAG System Ready")
    console.print("[dim]Type 'exit' or 'quit' to stop.[/dim]\n")

    while True:
        user_question = console.input("[bold magenta][?] Enter your question:[/bold magenta] ").strip()

        if user_question.lower() in ["exit", "quit"]:
            console.print("[dim][*] Exiting the RAG pipeline. Goodbye![/dim]")
            break

        if not user_question:
            console.print("[red][-] Please enter a valid question.[/red]")
            continue

        console.print()
        console.print(Panel(user_question, title="Question", border_style="cyan", expand=False))

        retrieved_chunks = agent.ask_question(user_question)
        chunk_texts = [c["text"] for c in retrieved_chunks]

        console.print()
        answer = agent.generate_answer(user_question, chunk_texts)

        if answer == "UNANSWERABLE":
            console.print(Panel(answer, title="Answer", border_style="yellow", expand=False))
        elif answer == "GENERATION_ERROR":
            console.print(Panel(answer, title="Answer", border_style="red", expand=False))
        else:
            console.print(Panel(answer, title="Answer", border_style="green", expand=False))

        console.print()


if __name__ == "__main__":
    main()
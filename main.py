from rag_engine.agent.rag_agent import RAGAgent
from visualize import visualize_embeddings

def main():
    print("[*] Starting the RAG pipeline...")
    
    agent = RAGAgent()
    
    print("\n[*] Beginning data ingestion process...")
    agent.ingest_data(None)  # None for default data path
    
    print("\n[+] Setup and Ingestion finished successfully!")

    visualize_embeddings()
    print("[+] Visualizer launched in your browser.")

    print("\n" + "="*50)
    print("RAG System is ready! Ask your questions.")
    print("Type 'exit' or 'quit' to stop.")
    print("="*50 + "\n")

    while True:
        user_question = input("\n[?] Enter your question: ").strip()
        
        if user_question.lower() in ['exit', 'quit']:
            print("[*] Exiting the RAG pipeline. Goodbye!")
            break
            
        if not user_question:
            print("[-] Please enter a valid question.")
            continue
            
        agent.ask_question(user_question)

if __name__ == "__main__":
    main()
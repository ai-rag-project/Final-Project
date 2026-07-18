import os
import numpy as np
from rag_engine.embedding.embedder import LocalEmbedder

def calculate_cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return dot_product / (norm_a * norm_b)

def main():
    file_path = "test_data.txt"
    
    if not os.path.exists(file_path):
        print(f"[-] Error: '{file_path}' not found in the root directory.")
        return

    embedder = LocalEmbedder()
    
    print("[*] Processing and embedding the document...")
    # فراخوانی متدی که در پیام قبلی ساختیم
    data = embedder.process_and_embed_file(file_path)
    
    chunks = data["chunks"]
    doc_embeddings = data["embeddings"]
    
    print(f"[+] Successfully loaded {len(chunks)} chunks into memory.\n")
    print("="*50)
    print(" Type ':q' to quit.")
    print("="*50)

    while True:
        query = input("\n[?] Enter your search query: ").strip()
        
        if query == ":q":
            print("[*] Goodbye!")
            break
            
        if not query:
            continue

        query_embedding = embedder.embed_query(query)

        similarities = []
        for i, doc_emb in enumerate(doc_embeddings):
            sim_score = calculate_cosine_similarity(query_embedding, doc_emb)
            similarities.append((sim_score, chunks[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        print("\n[+] Top 3 Results:")
        print("-" * 50)
        
        top_k = min(7, len(similarities))
        for i in range(top_k):
            score, text = similarities[i]

            percentage = round(score * 100, 2)
            
            print(f"Rank {i+1} | Relevance: {percentage}%")
            print(f"Chunk Text: {text}")
            print("-" * 50)

if __name__ == "__main__":
    main()
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

from rag_engine.vector_store.repository import VectorRepository

def visualize_embeddings():
    print("[*] Connecting to ChromaDB...")
    db = VectorRepository()
    
    collection_data = db.get_all_items()
    
    embeddings = collection_data.get("embeddings")
    documents = collection_data.get("documents")
    metadatas = collection_data.get("metadatas")
    
    if embeddings is None or len(embeddings) == 0:
        print("[-] No embeddings found in the database. Run ingestion first!")
        return

    print(f"[*] Retrieved {len(embeddings)} chunks. Reducing dimensions with t-SNE...")

    # Reduce dimensions from N to 2 using t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=10)
    reduced_vectors = tsne.fit_transform(embeddings)

    # Limit the 'text' to first 100 chars
    df = pd.DataFrame({
        'x': reduced_vectors[:, 0],
        'y': reduced_vectors[:, 1],
        'text': [doc[:100] + "..." for doc in documents],
        'context_id': [str(meta.get("context_id", "Unknown")) for meta in metadatas]
    })

    print("[*] Generating interactive plot...")

    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='context_id',
        hover_data=['text'],
        title="RAG Semantic Vector Space (t-SNE Projection)",
        labels={"context_id": "Document ID"}
    )
    
    fig.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()
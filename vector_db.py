import os
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.chunker import generate_chunks

# Vector Store Directory
DB_DIR = Path("data/chroma_db")

def get_embedding_function():
    """Returns CPU-optimized HuggingFace embedding model."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

def build_or_load_vector_db(force_rebuild: bool = False):
    """
    Builds a new ChromaDB index from document chunks if it doesn't exist,
    or loads the existing index from disk.
    """
    embedding_fn = get_embedding_function()

    if force_rebuild or not DB_DIR.exists():
        print("\n--- Generating Embeddings & Building ChromaDB Index ---")
        chunks = generate_chunks()
        
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_fn,
            persist_directory=str(DB_DIR)
        )
        print(f"Indexed {len(chunks)} chunks into ChromaDB at '{DB_DIR}'.")
    else:
        print("\n--- Loading Existing ChromaDB Vector Store ---")
        vector_db = Chroma(
            persist_directory=str(DB_DIR),
            embedding_function=embedding_fn
        )

    return vector_db

if __name__ == "__main__":
    # Test vector store creation and basic search
    db = build_or_load_vector_db(force_rebuild=True)
    
    # Query test
    query = "What is deep learning?"
    results = db.similarity_search(query, k=2)
    
    print("\n--- Test Query Results ---")
    for idx, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        print(f"Result {idx} (Source: {source}):\n{doc.page_content[:150]}...\n")
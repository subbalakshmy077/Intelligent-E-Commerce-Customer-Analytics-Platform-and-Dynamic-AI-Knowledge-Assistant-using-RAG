import os
import pandas as pd
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def generate_chunks(cleaned_dir_str: str = "data/cleaned", chunk_size: int = 500, chunk_overlap: int = 50) -> list[Document]:
    """
    Reads all cleaned text files and converts them into LangChain Document chunks with metadata attached.
    """
    cleaned_dir = Path(cleaned_dir_str)
    metadata_path = cleaned_dir.parent / "metadata.csv"
    
    meta_dict = {}
    if metadata_path.exists():
        df_meta = pd.read_csv(metadata_path)
        for _, row in df_meta.iterrows():
            meta_dict[row["File Name"]] = {
                "source": row["Article Name"],
                "url": row["URL"],
                "category": row.get("Category", "General")
            }

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )

    documents = []
    for file_path in cleaned_dir.glob("*.txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        file_metadata = meta_dict.get(file_path.name, {"source": file_path.stem})
        
        file_docs = text_splitter.create_documents(
            texts=[content],
            metadatas=[file_metadata]
        )
        documents.extend(file_docs)

    print(f"Generated {len(documents)} total chunks across all articles.")
    return documents

if __name__ == "__main__":
    docs = generate_chunks()
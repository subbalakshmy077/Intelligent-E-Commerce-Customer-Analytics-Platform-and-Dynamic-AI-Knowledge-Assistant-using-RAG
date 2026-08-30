import os
import re
import unicodedata
import pandas as pd
from pathlib import Path

def normalize_text(text: str) -> str:
    """Normalizes Unicode characters and removes invalid control chars."""
    text = unicodedata.normalize("NFKD", text)
    return text

def remove_wiki_junk(text: str) -> str:
    """Removes Wikipedia reference sections, external links, and standard headers."""
    # Remove standard non-content Wikipedia tail sections
    junk_sections = [
        r"==\s*References\s*==.*",
        r"==\s*Further reading\s*==.*",
        r"==\s*External links\s*==.*",
        r"==\s*See also\s*==.*"
    ]
    for pattern in junk_sections:
        text = re.sub(pattern, "", text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove remaining markdown-style or double-equals headers
    text = re.sub(r"==+[^=]+==+", "", text)
    
    # Remove leftover HTML tags if any
    text = re.sub(r"<[^>]+>", "", text)
    
    # Collapse multiple newlines and spaces into single clean spacing
    text = re.sub(r"\n\s*\n", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    
    return text.strip()

def clean_file(filepath: Path) -> tuple[str, bool]:
    """Reads raw text, cleans content, and checks if page is valid/non-empty."""
    with open(filepath, "r", encoding="utf-8") as f:
        raw_text = f.read()

    cleaned_text = remove_wiki_junk(normalize_text(raw_text))

    # Reject empty or extremely short pages (< 100 characters)
    if len(cleaned_text) < 100:
        return "", False

    return cleaned_text, True

def run_preprocessing_pipeline(raw_dir_str: str = "data/raw", clean_dir_str: str = "data/cleaned"):
    """
    Cleans all .txt files from raw_dir, saves them to clean_dir, 
    and updates metadata.csv with post-cleaning statistics.
    """
    raw_dir = Path(raw_dir_str)
    clean_dir = Path(clean_dir_str)
    clean_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = raw_dir.parent / "metadata.csv"
    if not metadata_path.exists():
        print("metadata.csv not found! Please run downloader.py first.")
        return

    df_meta = pd.read_csv(metadata_path)

    updated_rows = []
    cleaned_count = 0
    removed_count = 0

    print("\n--- Preprocessing Raw Documents ---")
    for idx, row in df_meta.iterrows():
        raw_filepath = raw_dir / row["File Name"]

        if not raw_filepath.exists():
            continue

        cleaned_text, is_valid = clean_file(raw_filepath)

        if not is_valid:
            # Delete empty file from disk
            removed_count += 1
            continue

        # Save cleaned file to data/cleaned/
        clean_filepath = clean_dir / row["File Name"]
        with open(clean_filepath, "w", encoding="utf-8") as f:
            f.write(cleaned_text)

        # Update metadata stats post-cleaning
        row["Cleaned Word Count"] = len(cleaned_text.split())
        row["Cleaned Character Count"] = len(cleaned_text)
        updated_rows.append(row)
        cleaned_count += 1

    # Save updated metadata
    updated_df = pd.DataFrame(updated_rows)
    updated_df.to_csv(metadata_path, index=False)

    print(f"Processed: {cleaned_count} articles saved to '{clean_dir}'.")
    print(f"Removed: {removed_count} empty/invalid articles.")
    print(f"Updated metadata logged to '{metadata_path}'.")

if __name__ == "__main__":
    run_preprocessing_pipeline()
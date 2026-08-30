import os
import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

HEADERS = {
    "User-Agent": "AIML-RAG-Dataset-Builder/1.0 (Contact: student@example.com)"
}
MAX_RETRIES = 3
SLEEP_TIME = 0.5

def clean_filename(name: str) -> str:
    """Removes invalid OS filename characters."""
    invalid = '<>:"/\\|?*'
    for ch in invalid:
        name = name.replace(ch, "_")
    return name

def search_articles(keyword: str, limit: int = 5) -> list:
    """Searches Wikipedia API for relevant article titles."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": keyword,
        "srlimit": limit,
        "format": "json"
    }
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    return [article["title"] for article in data["query"]["search"]]

def download_article(title: str) -> dict:
    """Downloads plain text extract, URL, and primary category for a title."""
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "redirects": 1,
        "prop": "extracts|info|categories",
        "inprop": "url",
        "cllimit": 1,
        "explaintext": 1
    }
    response = requests.get(url, params=params, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = response.json()
    
    page = list(data["query"]["pages"].values())[0]
    if "missing" in page:
        return None

    categories = page.get("categories", [])
    primary_category = categories[0]["title"].replace("Category:", "") if categories else "General"

    return {
        "title": page["title"],
        "text": page.get("extract", ""),
        "url": page.get("fullurl", ""),
        "category": primary_category
    }

def run_knowledge_builder(keywords: list, search_limit: int = 5, output_dir_str: str = "data/raw"):
    """Fetches articles for keywords and exports metadata.csv."""
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = []
    errors = []
    visited = set()

    candidate_titles = []
    print("\n--- Searching Wikipedia Titles ---")
    for keyword in keywords:
        try:
            results = search_articles(keyword, limit=search_limit)
            candidate_titles.extend(results)
        except Exception as e:
            print(f"Search failed for '{keyword}': {e}")

    candidate_titles = sorted(set(candidate_titles))
    print(f"Total Unique Articles Identified: {len(candidate_titles)}")

    print("\n--- Downloading Articles ---")
    for title in tqdm(candidate_titles):
        if title in visited:
            continue
        visited.add(title)

        for retry in range(MAX_RETRIES):
            try:
                page = download_article(title)
                if page is None:
                    raise Exception("Page missing")

                filename = clean_filename(page["title"]) + ".txt"
                filepath = output_dir / filename
                text_content = page["text"]

                if not filepath.exists():
                    with open(filepath, "w", encoding="utf-8") as f:
                        f.write(text_content)

                metadata.append({
                    "Article Name": page["title"],
                    "URL": page["url"],
                    "Category": page["category"],
                    "Word Count": len(text_content.split()),
                    "Character Count": len(text_content),
                    "Download Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "File Name": filename
                })
                break

            except Exception as e:
                if retry == MAX_RETRIES - 1:
                    errors.append({"Title": title, "Error": str(e)})
                time.sleep(SLEEP_TIME)

    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(output_dir.parent / "metadata.csv", index=False)

    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(output_dir.parent / "error_log.csv", index=False)

    return metadata_df, errors_df

if __name__ == "__main__":
    sample_keywords = [
        "Artificial Intelligence", "Machine Learning", 
        "Deep Learning", "Large Language Model", "Natural Language Processing"
    ]
    meta, err = run_knowledge_builder(sample_keywords, search_limit=3)
    print(f"\nCompleted: {len(meta)} articles saved in data/raw/ and logged to data/metadata.csv.")
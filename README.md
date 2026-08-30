# 🛍️ Intelligent E-Commerce Customer Analytics Platform

## Overview
An end-to-end retail intelligence platform built on the Brazilian E-Commerce (Olist) dataset[cite: 3]. The system merges multi-table relational transactional data to perform RFM customer segmentation, predict 90-day repeat purchases, forecast Customer Lifetime Value (CLV), and interpret predictions using Explainable AI (XAI)[cite: 3].

## Key Features
* **Data Integration & Cleaning:** Joins relational tables across orders, customers, payments, products, and reviews[cite: 3].
* **RFM Customer Segmentation:** Categorizes customers into behavioral segments using K-Means clustering[cite: 3].
* **Predictive Analytics:** Features Random Forest models for 90-day repeat purchase classification and CLV regression[cite: 3].
* **Explainable AI (XAI):** Visualizes feature importance and SHAP plots for decision transparency[cite: 3].
* **Interactive Dashboard:** Built using Streamlit for real-time predictions and executive metric tracking[cite: 3].

## Tech Stack
* Python, Pandas, NumPy, Scikit-Learn, Random Forest, XGBoost, SHAP, Matplotlib, Streamlit[cite: 3]

## Setup & Execution
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt

###: Dynamic RAG Assistant

```markdown
# 🧠 Dynamic AI Knowledge Assistant using RAG

## Overview
A dynamic, context-aware Generative AI assistant that builds its own vector knowledge base from online sources without manual document collection[cite: 2]. Given user-defined keywords, the application fetches relevant Wikipedia articles, chunks and embeds the content, stores vectors in ChromaDB, and generates grounded answers using an advanced RAG pipeline with direct source citations[cite: 2].

## Key Features
* **Automated Data Acquisition:** Fetches and preprocesses Wikipedia articles dynamically based on input keywords[cite: 2].
* **Vector Indexing:** Processes text chunks and indexes embeddings in persistent vector stores (ChromaDB/FAISS)[cite: 2].
* **RAG Pipeline & Failover:** Retrieves semantic context and routes prompts to active LLMs (Groq / Gemini) with automated failover logic[cite: 2].
* **Source Attribution:** Displays exact article sources and context chunks alongside generated responses for transparency[cite: 2].

## Tech Stack
* Python, LangChain, Sentence-Transformers, ChromaDB, Wikipedia API, Groq API, Google Gemini API, Streamlit[cite: 2]

## Setup & Execution
1. Clone the repository and install dependencies:
   ```bash
   pip install -r requirements.txt

import os
import sys
import re
import json
from pathlib import Path
import os
from groq import Groq
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI

# Fix system path to resolve 'src' imports cleanly across environments
FILE_PATH = Path(__file__).resolve()
ROOT_DIR = FILE_PATH.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from src.vector_db import build_or_load_vector_db

# Load API keys from .env at project root
load_dotenv(ROOT_DIR / ".env", override=True)


def extract_clean_text(response) -> str:
    """
    Extracts pure text out of raw strings, lists, dicts, or AIMessage objects.
    Strips away metadata dictionaries containing 'extras', 'signature', or 'type'.
    """
    # 1. Access content attribute if passed an AIMessage or ChatResult object
    content = getattr(response, "content", response)

    text_parts = []

    # 2. Extract plain text strings from list blocks or dicts
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and "text" in block:
                text_parts.append(str(block["text"]))
            elif hasattr(block, "text"):
                text_parts.append(str(block.text))
            elif isinstance(block, str):
                text_parts.append(block)
        raw_text = "\n".join(text_parts) if text_parts else str(content)
    elif isinstance(content, dict) and "text" in content:
        raw_text = str(content["text"])
    else:
        raw_text = str(content)

    # 3. Regex cleanup to strip trailing metadata dicts ('extras', 'signature')
    cleaned_text = re.sub(
        r"\{'type':.*?\}|\{'extras':.*?\}|\{'signature':.*?\}",
        "",
        raw_text,
        flags=re.DOTALL,
    )

    # 4. Fallback string splitting for hardcoded metadata representations
    if "'extras':" in cleaned_text:
        cleaned_text = cleaned_text.split("'extras':")[0]
    if "'signature':" in cleaned_text:
        cleaned_text = cleaned_text.split("'signature':")[0]

    return cleaned_text.strip().rstrip(",'\" {}[]")





def get_llm(provider: str = "groq"):
    """
    Initializes and returns an active LLM instance by dynamically fetching 
    available endpoints from Groq or targeting supported Gemini models.
    """
    provider_clean = provider.lower().strip()

    if provider_clean == "groq":
        raw_key = os.getenv("GROQ_API_KEY", "")
        api_key = raw_key.strip().strip('"').strip("'")
        
        if not api_key:
            raise ValueError("GROQ_API_KEY is missing or empty in your .env file!")
        
        try:
            # Dynamically fetch available models directly from Groq's API
            client = Groq(api_key=api_key)
            available_models = [m.id for m in client.models.list().data if "llama" in m.id or "mixtral" in m.id]
            
            # Use the first active model found, fallback to standard default
            active_model = available_models[0] if available_models else "llama-3.3-70b-versatile"
            
            return ChatGroq(
                groq_api_key=api_key,
                model_name=active_model,
                temperature=0.2
            )
        except Exception as e:
            raise RuntimeError(f"Failed to fetch active models from Groq API: {e}")

    elif provider_clean == "gemini":
        raw_key = os.getenv("GEMINI_API_KEY", "")
        api_key = raw_key.strip().strip('"').strip("'")
        
        if not api_key:
            raise ValueError("GEMINI_API_KEY is missing or empty in your .env file!")
            
        # Updated to active production model endpoint
        return ChatGoogleGenerativeAI(
            google_api_key=api_key,
            model="gemini-3.6-flash",
            temperature=0.2
        )

    else:
        raise ValueError("Invalid provider specified. Use 'groq' or 'gemini'.")
    
RAG_PROMPT_TEMPLATE = """You are an expert AI Assistant answering questions based strictly on provided source documents.

Context Information:
{context}

Question: 
{question}

Instructions:
1. Answer the question using ONLY the provided context above.
2. If the context does not contain enough information to answer, state clearly: "I cannot find sufficient information in the knowledge base."
3. Cite the source document names at the end of your answer.

Answer:
"""


def answer_query(query: str, provider: str = "groq", k: int = 3) -> dict:
    """
    Retrieves top k matching chunks from ChromaDB, constructs the prompt, 
    and returns the LLM response along with source metadata. Includes automated model fallbacks.
    """
    # 1. Load Vector Database & Retrieve Context
    vector_db = build_or_load_vector_db(force_rebuild=False)
    retrieved_docs = vector_db.similarity_search(query, k=k)

    if not retrieved_docs:
        return {
            "answer": "No relevant documents found in the vector store.",
            "sources": [],
            "context_chunks": []
        }

    # 2. Format Context & Collect Sources
    context_text = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
    sources = list(set([doc.metadata.get("source", "Unknown") for doc in retrieved_docs]))

    # 3. Construct Prompt Template
    prompt = PromptTemplate(
        template=RAG_PROMPT_TEMPLATE,
        input_variables=["context", "question"]
    )
    formatted_prompt = prompt.format(context=context_text, question=query)

    # 4. Invoke LLM with Fallback Support
    try:
        llm = get_llm(provider=provider)
        response = llm.invoke(formatted_prompt)
        answer_text = extract_clean_text(response)
    except Exception as primary_error:
        print(f"Warning: Primary provider ({provider}) failed: {primary_error}")
        
        # Automatic Fallback Routine
        fallback_provider = "gemini" if provider.lower() == "groq" else "groq"
        print(f"Attempting fallback to provider: {fallback_provider}...")
        
        try:
            fallback_llm = get_llm(provider=fallback_provider)
            response = fallback_llm.invoke(formatted_prompt)
            answer_text = extract_clean_text(response)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Both primary ({provider}) and fallback ({fallback_provider}) LLM calls failed.\n"
                f"Primary Error: {primary_error}\n"
                f"Fallback Error: {fallback_error}"
            )

    return {
        "answer": answer_text,
        "sources": sources,
        "context_chunks": [doc.page_content for doc in retrieved_docs]
    }


if __name__ == "__main__":
    test_query = "What is artificial intelligence and deep learning?"
    print(f"\n--- Querying RAG Engine: '{test_query}' ---")
    
    # Run query test via Groq
    result = answer_query(test_query, provider="groq", k=3)
    
    print("\n--- Generated Answer ---")
    print(result["answer"])
    print("\n--- Cited Sources ---")
    for src in result["sources"]:
        print(f"- {src}")
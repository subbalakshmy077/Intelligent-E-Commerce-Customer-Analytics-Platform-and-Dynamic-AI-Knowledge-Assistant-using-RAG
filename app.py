import os
import sys
import time
import pandas as pd
import plotly.express as px
import streamlit as st
from pathlib import Path

# Add project root to sys.path for clean imports
ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from src.downloader import run_knowledge_builder
from src.preprocessor import run_preprocessing_pipeline
from src.vector_db import build_or_load_vector_db
from src.rag_engine import answer_query

# Configure Streamlit Page Layout
st.set_page_config(
    page_title="Dynamic AI Knowledge Assistant",
    page_icon="🤖",
    layout="wide"
)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", [
    "🤖 Chat Assistant",
    "📥 Dataset Builder",
    "📚 Knowledge Base",
    "📊 Analytics Dashboard"
])

METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"

# ==========================================
# PAGE 1: CHAT ASSISTANT
# ==========================================
if page == "🤖 Chat Assistant":
    st.title("🤖 Dynamic RAG Assistant")
    st.write("Query your vector-indexed Wikipedia knowledge base with real-time source citations.")

    col1, col2 = st.columns([3, 1])
    with col2:
        provider = st.selectbox("LLM Provider", ["groq", "gemini"])
        top_k = st.slider("Retrieved Chunks (k)", 1, 5, 3)

    # Initialize Chat History
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display Chat History
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User Input & RAG Chain Invocation
    if user_query := st.chat_input("Ask a question based on your indexed articles..."):
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching ChromaDB and generating answer..."):
                start_time = time.time()
                response = answer_query(user_query, provider=provider, k=top_k)
                latency = time.time() - start_time

                st.markdown(response["answer"])
                
                if response["sources"]:
                    st.caption(f"**Sources Cited:** {', '.join(response['sources'])} | ⚡ Response Time: {latency:.2f}s")
                
                with st.expander("🔍 View Retrieved Vector Context Chunks"):
                    for idx, chunk in enumerate(response["context_chunks"], 1):
                        st.markdown(f"**Chunk {idx}:**\n{chunk}\n")

        st.session_state.messages.append({
            "role": "assistant", 
            "content": response["answer"]
        })

# ==========================================
# PAGE 2: DATASET BUILDER
# ==========================================
elif page == "📥 Dataset Builder":
    st.title("📥 Automated Knowledge Base Builder")
    st.write("Fetch new domain articles from Wikipedia API, clean text, and rebuild ChromaDB vector store.")

    keywords_input = st.text_area("Seed Keywords (comma-separated)", "Artificial Intelligence, Machine Learning, Deep Learning")
    search_limit = st.number_input("Articles per keyword", min_value=1, max_value=10, value=3)

    if st.button("Fetch & Build Knowledge Base"):
        keywords = [k.strip() for k in keywords_input.split(",") if k.strip()]
        
        st.info("Step 1/3: Downloading Wikipedia articles...")
        meta_df, err_df = run_knowledge_builder(keywords, search_limit=search_limit)
        
        st.info("Step 2/3: Preprocessing and cleaning text...")
        run_preprocessing_pipeline()

        st.info("Step 3/3: Rebuilding ChromaDB vector store...")
        build_or_load_vector_db(force_rebuild=True)

        st.success(f"Successfully indexed {len(meta_df)} articles into ChromaDB!")

# ==========================================
# PAGE 3: KNOWLEDGE BASE OVERVIEW
# ==========================================
elif page == "📚 Knowledge Base":
    st.title("📚 Knowledge Base Overview")

    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Total Articles", len(df))
        m2.metric("Total Raw Words", df["Word Count"].sum())
        m3.metric("Total Cleaned Words", df.get("Cleaned Word Count", df["Word Count"]).sum())

        st.dataframe(df, use_container_width=True)
    else:
        st.warning("No metadata found. Use the Dataset Builder page to generate your dataset.")

# ==========================================
# PAGE 4: ANALYTICS DASHBOARD
# ==========================================
elif page == "📊 Analytics Dashboard":
    st.title("📊 Dataset Analytics")

    if METADATA_PATH.exists():
        df = pd.read_csv(METADATA_PATH)

        col1, col2 = st.columns(2)
        with col1:
            fig_bar = px.bar(
                df, 
                x="Article Name", 
                y="Word Count", 
                title="Word Count per Article", 
                color="Category"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col2:
            fig_pie = px.pie(
                df, 
                names="Category", 
                title="Articles Distribution by Category"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.warning("No metadata found to generate analytics charts.")
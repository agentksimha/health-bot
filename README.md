#  Health Chatbot — Retrieval-Augmented Generation (RAG) with FAISS and Async LLM

An intelligent, **Retrieval-Augmented Generation chatbot** built for health awareness and disease prevention.  
It combines **semantic document retrieval (FAISS)**, **asynchronous Together API calls**, and a **Streamlit interface** — deployed in Docker.

---

## Features
- **FAISS vector search** for semantic retrieval from medical PDFs.  
- **Async LLM generation** using Together API (DeepSeek-V3).  
- **Live health news** via asynchronous RSS fetching.  
- **Multiple chat sessions** with context history.  
- **Benchmarking tool** for FAISS retrieval speed.  
- **Dockerized deployment** for full reproducibility.

---

## Tech Stack
**Languages:** Python  
**Libraries:** Streamlit, FAISS, SentenceTransformers, aiohttp, feedparser, huggingface_hub, Together API  
**Containerization:** Docker

---

## Installation

### Clone the repository
```bash
git clone https://github.com/agentksimha/health-bot.git
cd health-bot

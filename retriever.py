import faiss
import pickle
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download
import numpy as np

class Retriever:
    """RAG Retriever using FAISS and SentenceTransformer embeddings."""

    def __init__(self):
        self.index = None
        self.metadata = None
        self.embed_model = None
        self.load_index()

    def load_index(self):
        """Load FAISS index, metadata, and embedding model."""
        faiss_path = hf_hub_download(
            repo_id="krishnasimha/health-chatbot-data",
            filename="health_index.faiss",
            repo_type="dataset"
        )
        pkl_path = hf_hub_download(
            repo_id="krishnasimha/health-chatbot-data",
            filename="health_metadata.pkl",
            repo_type="dataset"
        )

        self.index = faiss.read_index(faiss_path)
        with open(pkl_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    def retrieve(self, query, k=3):
        """Retrieve top-k passages and their sources for a query."""
        query_emb = self.embed_model.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_emb, k)
        retrieved = [self.metadata["texts"][i] for i in I[0]]
        sources = [self.metadata["sources"][i] for i in I[0]]
        context = "\n".join(retrieved)
        return context, sources

import numpy as np
import time

def benchmark_faiss(retriever, n_queries=100, k=3):
    """
    Benchmark FAISS retrieval speed.
    
    Args:
        retriever (Retriever): instance of Retriever class
        n_queries (int): number of queries
        k (int): top-k results
    """
    queries = ["What is diabetes?", "How to prevent malaria?", "Symptoms of dengue?"]
    query_embs = retriever.embed_model.encode(queries, convert_to_numpy=True)

    times = []
    for _ in range(n_queries):
        q = query_embs[np.random.randint(0, len(query_embs))].reshape(1, -1)
        start = time.time()
        retriever.index.search(q, k)
        times.append(time.time() - start)

    avg_time = np.mean(times) * 1000
    return avg_time

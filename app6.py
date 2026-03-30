# app.py
import os
import time
import datetime
import asyncio
import sqlite3
import pickle
import re
import aiohttp

import streamlit as st
import numpy as np
import pandas as pd
import feedparser

from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import hf_hub_download

# Optional S3 support
try:
    import boto3
    BOTO3_AVAILABLE = True
except Exception:
    BOTO3_AVAILABLE = False

import faiss

# -------------------------
# Initialize DB & helpers
# -------------------------
DB_PATH = "query_cache.db"

def init_cache_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT UNIQUE,
            answer TEXT,
            embedding BLOB,
            frequency INTEGER DEFAULT 1
        )
    """)
    conn.commit()
    conn.close()

def init_export_logs():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS export_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            exported_on TEXT,
            file_name TEXT
        )
    """)
    conn.commit()
    conn.close()

def init_metrics_log():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS metrics_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            query TEXT,
            answer_len INTEGER,
            sources_count INTEGER,
            confidence REAL
        )
    """)
    conn.commit()
    conn.close()

init_cache_db()
init_export_logs()
init_metrics_log()

def get_db_connection():
    return sqlite3.connect(DB_PATH)

# -------------------------
# Cache store/search
# -------------------------
def store_in_cache(query, answer, embedding):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT OR REPLACE INTO cache (query, answer, embedding, frequency)
        VALUES (?, ?, ?, COALESCE(
            (SELECT frequency FROM cache WHERE query=?), 0
        ) + 1)
    """, (query, answer, embedding.tobytes(), query))
    conn.commit()
    conn.close()

def search_cache(query, embed_model, threshold=0.85):
    q_emb = embed_model.encode([query], convert_to_numpy=True)[0]
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("SELECT query, answer, embedding, frequency FROM cache")
    rows = c.fetchall()
    conn.close()

    best_sim = -1
    best_row = None

    for qry, ans, emb_blob, freq in rows:
        try:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
        except Exception:
            continue
        emb = emb.reshape(-1)
        sim = np.dot(q_emb, emb) / (np.linalg.norm(q_emb) * np.linalg.norm(emb) + 1e-12)
        if sim > threshold and sim > best_sim:
            best_sim = sim
            best_row = (qry, ans, freq)

    if best_row:
        return best_row[1]
    return None

# -------------------------
# Metrics logging
# -------------------------
def log_metrics(query, answer_len, sources_count, confidence):
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO metrics_log (timestamp, query, answer_len, sources_count, confidence)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.datetime.now().isoformat(), query, answer_len, sources_count, confidence))
    conn.commit()
    conn.close()

# -------------------------
# Exports
# -------------------------
def export_cache_to_excel():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT * FROM cache", conn)
    conn.close()
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"cache_export_{timestamp}.xlsx"
    df.to_excel(file_name, index=False)
    # log export
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO export_logs (exported_on, file_name) VALUES (?, ?)",
              (datetime.datetime.now().isoformat(), file_name))
    conn.commit()
    conn.close()
    return file_name

def export_cache_to_sql():
    conn = get_db_connection()
    dump_path = f"cache_dump_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
    with open(dump_path, "w", encoding="utf-8") as f:
        for line in conn.iterdump():
            f.write("%s\n" % line)
    conn.close()
    # log export
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("INSERT INTO export_logs (exported_on, file_name) VALUES (?, ?)",
              (datetime.datetime.now().isoformat(), dump_path))
    conn.commit()
    conn.close()
    return dump_path

# -------------------------
# Optional: S3 upload
# -------------------------
def upload_file_to_s3(local_path, bucket_name, object_name=None):
    if not BOTO3_AVAILABLE:
        return False, "boto3 not installed"
    if object_name is None:
        object_name = os.path.basename(local_path)
    try:
        s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            region_name=os.environ.get("AWS_DEFAULT_REGION")
        )
        s3.upload_file(local_path, bucket_name, object_name)
        return True, f"s3://{bucket_name}/{object_name}"
    except Exception as e:
        return False, str(e)

# -------------------------
# Load FAISS index
# -------------------------
@st.cache_resource
def load_index():
    faiss_path = hf_hub_download("krishnasimha/health-chatbot-data", "health_index.faiss", repo_type="dataset")
    pkl_path = hf_hub_download("krishnasimha/health-chatbot-data", "health_metadata.pkl", repo_type="dataset")
    index = faiss.read_index(faiss_path)
    with open(pkl_path, "rb") as f:
        metadata = pickle.load(f)
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, metadata, embed_model

index, metadata, embed_model = load_index()

# -------------------------
# Load Reranker (Cross-Encoder)
# -------------------------
@st.cache_resource
def load_reranker():
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

reranker = load_reranker()

# -------------------------
# FAISS benchmark
# -------------------------
def benchmark_faiss(n_queries=100, k=3):
    queries = ["What is diabetes?", "How to prevent malaria?", "Symptoms of dengue?"]
    query_embs = embed_model.encode(queries, convert_to_numpy=True)
    times = []
    for _ in range(n_queries):
        q = query_embs[np.random.randint(0, len(query_embs))].reshape(1, -1)
        start = time.time()
        D, I = index.search(q, k)
        times.append(time.time() - start)
    avg_time = np.mean(times) * 1000
    st.sidebar.write(f"⚡ FAISS Benchmark: {avg_time:.2f} ms/query over {n_queries} queries")

# -------------------------
# RSS / News
# -------------------------
RSS_URL = "https://news.google.com/rss/search?q=health+disease+awareness&hl=en-IN&gl=IN&ceid=IN:en"

async def fetch_rss_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as resp:
            return await resp.text()

def fetch_news():
    try:
        raw_xml = asyncio.run(fetch_rss_url(RSS_URL))
        feed = feedparser.parse(raw_xml)
        articles = [{"title": e.get("title",""), "link": e.get("link",""), "published": e.get("published","")} for e in feed.entries[:5]]
        return articles
    except Exception:
        return []

def update_news_hourly():
    now = datetime.datetime.now()
    if "last_news_update" not in st.session_state or (now - st.session_state.last_news_update).seconds > 3600:
        st.session_state.last_news_update = now
        st.session_state.news_articles = fetch_news()

# -------------------------
# Together API
# -------------------------
async def async_together_chat(messages):
    url = "https://api.together.xyz/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.environ.get('TOGETHER_API_KEY','')}",
        "Content-Type": "application/json",
    }
    payload = {"model": "deepseek-ai/DeepSeek-V3", "messages": messages}
    async with aiohttp.ClientSession() as session:
        async with session.post(url, headers=headers, json=payload) as resp:
            result = await resp.json()
            return result["choices"][0]["message"]["content"]

# -------------------------
# Corrective Web Search
# -------------------------
async def web_search_corrective(query, n_results=3):
    prompt = f"Provide top {n_results} factual paragraphs about: {query}"
    async with aiohttp.ClientSession() as session:
        async with session.post(
            "https://api.together.xyz/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ['TOGETHER_API_KEY']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-ai/DeepSeek-V3",
                "messages": [{"role":"user","content":prompt}]
            }
        ) as r:
            out = await r.json()
            text = out["choices"][0]["message"]["content"]
            return text.split("\n")[:n_results]

# -------------------------
# Confidence metric
# -------------------------
def answer_confidence(scores, threshold=0.7):
    if scores is None or len(scores) == 0:
        return False
    return float(np.max(scores)) >= threshold

# -------------------------
# Retrieve answer (CAG)
# -------------------------
def retrieve_answer_cag(query, k=3, confidence_threshold=0.7):
    cached_answer = search_cache(query, embed_model)
    if cached_answer:
        st.sidebar.success("⚡ Retrieved from cache")
        return cached_answer, [], 1.0

    query_emb = embed_model.encode([query], convert_to_numpy=True)
    fetch_k = max(k, 10)
    D, I = index.search(query_emb, fetch_k)

    retrieved = [metadata["texts"][i] for i in I[0]]
    sources = [metadata["sources"][i] for i in I[0]]

    try:
        pairs = [[query, chunk] for chunk in retrieved]
        scores = reranker.predict(pairs)
        reranked = sorted(zip(scores, retrieved, sources), key=lambda x: x[0], reverse=True)
        top_reranked = reranked[:k]
        top_chunks = [c for _, c, _ in top_reranked]
        top_sources = [s for _, _, s in top_reranked]
        context = "\n".join(top_chunks)
        sources = top_sources
    except Exception:
        context = "\n".join(retrieved[:k])
        sources = sources[:k]
        scores = [1.0] * k

    confident = answer_confidence(scores, threshold=confidence_threshold)
    corrective_context = []
    if not confident:
        st.sidebar.warning("⚠ Low confidence — fetching corrective evidence...")
        corrective_context = asyncio.run(web_search_corrective(query))
        context += "\n" + "\n".join(corrective_context)
        sources += ["Corrective Web"] * len(corrective_context)

    user_message = {"role":"user", "content": f"Answer based on context:\n{context}\n\nQuestion: {query}"}
    st.session_state.chats[st.session_state.current_chat].append(user_message)

    try:
        answer = asyncio.run(async_together_chat(st.session_state.chats[st.session_state.current_chat]))
    except Exception as e:
        answer = f"Error: {e}"

    try:
        store_in_cache(query, answer, query_emb[0])
    except Exception:
        pass

    st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": answer})
    confidence_score = float(np.max(scores)) if scores is not None and len(scores) > 0 else 0.0


    # -------------------------
    # Log metrics persistently
    # -------------------------
    log_metrics(query, len(answer), len(sources), confidence_score)

    return answer, sources, confidence_score

# -------------------------
# Streamlit UI + Chat
# -------------------------
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat 1"
    st.session_state.chats["New Chat 1"] = [{"role": "system", "content": "You are a helpful public health chatbot."}]

if "metrics_log" not in st.session_state:
    st.session_state.metrics_log = []

st.sidebar.header("Chat Manager")
if st.sidebar.button("➕ New Chat"):
    chat_count = len(st.session_state.chats) + 1
    new_chat_name = f"New Chat {chat_count}"
    st.session_state.chats[new_chat_name] = [{"role": "system", "content": "You are a helpful public health chatbot."}]
    st.session_state.current_chat = new_chat_name

benchmark_faiss()

chat_list = list(st.session_state.chats.keys())
selected_chat = st.sidebar.selectbox("Your chats:", chat_list, index=chat_list.index(st.session_state.current_chat))
st.session_state.current_chat = selected_chat

new_name = st.sidebar.text_input("Rename Chat:", st.session_state.current_chat)
if new_name and new_name != st.session_state.current_chat:
    if new_name not in st.session_state.chats:
        st.session_state.chats[new_name] = st.session_state.chats.pop(st.session_state.current_chat)
        st.session_state.current_chat = new_name

# -------------------------
# Admin Panel
# -------------------------
query_params = st.query_params
is_admin_mode = (query_params.get("admin") == ["1"])

def rerun_app():
    st.session_state['__rerun'] = not st.session_state.get('__rerun', False)

ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "")

if is_admin_mode or st.session_state.get("is_admin", False):
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔐 Admin Panel (dev only)")

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    # Login/logout
    if st.session_state.is_admin:
        st.sidebar.success("Admin authenticated")
        if st.sidebar.button("🚪 Logout Admin"):
            st.session_state.is_admin = False
            rerun_app()
    else:
        admin_input = st.sidebar.text_input("Enter admin password:", type="password")
        if st.sidebar.button("Login"):
            if admin_input == ADMIN_PASSWORD and ADMIN_PASSWORD != "":
                st.session_state.is_admin = True
                st.sidebar.success("Admin authenticated")
                rerun_app()
            else:
                st.sidebar.error("Wrong password or ADMIN_PASSWORD not set")

    # Admin features
    if st.session_state.is_admin:
        st.sidebar.markdown("### 📋 View User Queries")
        conn = get_db_connection()
        df_cache = pd.read_sql_query("SELECT query, answer, frequency FROM cache ORDER BY id DESC", conn)
        conn.close()
        st.sidebar.dataframe(df_cache)

        st.sidebar.markdown("### 📊 Persistent Metrics Log")
        conn = get_db_connection()
        df_metrics = pd.read_sql_query("SELECT * FROM metrics_log ORDER BY id DESC", conn)
        conn.close()
        st.sidebar.dataframe(df_metrics)

        st.sidebar.markdown("### ⬇️ Export Query Cache")
        if st.sidebar.button("Export to Excel"):
            file_path = export_cache_to_excel()
            st.sidebar.success(f"Exported: {file_path}")
            with open(file_path, "rb") as f:
                st.sidebar.download_button("Download Excel", f, file_name=file_path, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        if st.sidebar.button("Export to SQL"):
            file_path = export_cache_to_sql()
            st.sidebar.success(f"Exported: {file_path}")
            with open(file_path, "rb") as f:
                st.sidebar.download_button("Download SQL", f, file_name=file_path, mime="application/sql")

# -------------------------
# Main UI
# -------------------------
st.title(st.session_state.current_chat)
update_news_hourly()
st.subheader("📰 Latest Health Updates")
if "news_articles" in st.session_state:
    for art in st.session_state.news_articles:
        st.markdown(f"**{art['title']}**  \n[Read more]({art['link']})  \n*Published: {art['published']}*")
        st.write("---")

user_query = st.text_input("Ask me about health, prevention, or awareness:")

if user_query:
    with st.spinner("Searching knowledge base..."):
        answer, sources, confidence = retrieve_answer_cag(user_query)

    st.write("### 💡 Answer")
    st.write(answer)
    st.write(f"*Confidence Score: {confidence:.2f}*")

    st.write("### 📖 Sources")
    for src in sources:
        st.write(f"- {src}")

    # log metrics in session_state
    st.session_state.metrics_log.append({
        "query": user_query,
        "answer_len": len(answer),
        "sources_count": len(sources),
        "confidence": confidence
    })

# render chat history
for msg in st.session_state.chats[st.session_state.current_chat]:
    if msg["role"] == "user":
        st.write(f"🧑 **You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.write(f"🤖 **Bot:** {msg['content']}")

# show session metrics
if st.sidebar.checkbox("📊 Show Metrics"):
    st.sidebar.write(pd.DataFrame(st.session_state.metrics_log))


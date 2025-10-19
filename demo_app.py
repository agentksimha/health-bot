import streamlit as st
from retriever import Retriever
from llm_async import run_async_chat
from news_fetcher import update_news_hourly
from benchmark import benchmark_faiss

# Initialize retriever
retriever = Retriever()

# Session state setup
if "chats" not in st.session_state:
    st.session_state.chats = {}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat 1"
    st.session_state.chats["New Chat 1"] = [
        {"role": "system", "content": "You are a helpful public health awareness chatbot."}
    ]

# Sidebar: chat manager
st.sidebar.header("Chat Manager")
if st.sidebar.button("➕ New Chat"):
    chat_count = len(st.session_state.chats) + 1
    new_chat_name = f"New Chat {chat_count}"
    st.session_state.chats[new_chat_name] = [
        {"role": "system", "content": "You are a helpful public health awareness chatbot."}
    ]
    st.session_state.current_chat = new_chat_name

# Select chat
chat_list = list(st.session_state.chats.keys())
selected_chat = st.sidebar.selectbox("Your chats:", chat_list, index=chat_list.index(st.session_state.current_chat))
st.session_state.current_chat = selected_chat

# Rename chat
new_name = st.sidebar.text_input("Rename Chat:", st.session_state.current_chat)
if new_name and new_name != st.session_state.current_chat:
    if new_name not in st.session_state.chats:
        st.session_state.chats[new_name] = st.session_state.chats.pop(st.session_state.current_chat)
        st.session_state.current_chat = new_name

# FAISS benchmark
avg_time = benchmark_faiss(retriever)
st.sidebar.write(f"⚡ FAISS Benchmark: {avg_time:.2f} ms/query over 100 queries")

# Update news
update_news_hourly(st.session_state)
st.subheader("📰 Latest Health Updates")
if "news_articles" in st.session_state:
    for art in st.session_state.news_articles:
        st.markdown(f"**{art['title']}**  \n[Read more]({art['link']})  \n*Published: {art['published']}*")
        st.write("---")

# Chat input
user_query = st.text_input("Ask me about health, prevention, or awareness:")
if user_query:
    # Retrieve context
    context, sources = retriever.retrieve(user_query)
    user_message = {
        "role": "user",
        "content": f"Answer based on the context below:\n\n{context}\n\nQuestion: {user_query}"
    }
    st.session_state.chats[st.session_state.current_chat].append(user_message)

    # Run LLM
    answer = run_async_chat(st.session_state.chats[st.session_state.current_chat])
    st.session_state.chats[st.session_state.current_chat].append({"role": "assistant", "content": answer})

    # Display
    st.write("### 💡 Answer")
    st.write(answer)
    st.write("### 📖 Sources")
    for src in sources:
        st.write(f"- {src}")

# Display chat history
for msg in st.session_state.chats[st.session_state.current_chat]:
    if msg["role"] == "user":
        st.write(f"🧑 **You:** {msg['content']}")
    elif msg["role"] == "assistant":
        st.write(f"🤖 **Bot:** {msg['content']}")

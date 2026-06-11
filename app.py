import streamlit as st
import requests
from rag_core import extract_text, create_chunks, create_vector_store, retrieve_answer
 
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
 
 
def ask_groq(context: str, question: str, api_key: str) -> str:
 
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
 
    payload = {
        "model": "llama-3.1-8b-instant",
        "messages": [
            {
                "role": "user",
                "content": f"""Answer the question based only on the context below.
If the answer is not in the context, say "This information is not in the document."
 
Context:
{context}
 
Question: {question}
 
Answer:"""
            }
        ],
        "max_tokens": 300,
        "temperature": 0.3
    }
 
    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
 
        if response.status_code == 401:
            return "❌ Invalid API Key. Please check your Groq API key in the sidebar."
 
        if response.status_code == 429:
            return "⚠️ Rate limit exceeded. Please try again later."
 
        if response.status_code != 200:
            return f"❌ API Error {response.status_code}: {response.text[:200]}"
 
        return response.json()["choices"][0]["message"]["content"]
 
    except requests.exceptions.Timeout:
        return "⏳ Timeout. Please try again."
 
    except Exception as e:
        return f"❌ Error: {e}"
 
 
# -------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------
 
st.set_page_config(
    page_title="AI Study Buddy",
    page_icon="📚",
    layout="centered",
    initial_sidebar_state="expanded",
)
 
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
 
.chunk-box {
    background: #f8f9fa;
    border-left: 4px solid #4CAF50;
    padding: 10px 15px;
    margin: 8px 0;
    border-radius: 0 8px 8px 0;
    font-size: 0.85rem;
    color: #333;
}
 
.score-badge {
    background: #4CAF50;
    color: white;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 0.75rem;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)
 
 
# -------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------
 
with st.sidebar:
 
    st.header("⚙️ Settings")
 
    # --- Groq API Key Input ---
    st.markdown("**🔑 Groq API Key**")
 
    groq_api_key = st.text_input(
        "Enter your Groq API Key",
        type="password",
        placeholder="gsk_xxxxxxxxxxxxxxxx",
        help="Free key from console.groq.com — no credit card needed"
    )
 
    if not groq_api_key:
        st.warning("⚠️ API key required. Get free key from [console.groq.com](https://console.groq.com)")
    else:
        st.success("✅ API Key entered")
 
    st.divider()
 
    chunk_size = st.slider(
        "Chunk Size (characters)",
        200,
        1000,
        500,
        50,
        help="Smaller = More precise, Larger = More context"
    )
 
    overlap = st.slider(
        "Chunk Overlap (characters)",
        0,
        300,
        100,
        25,
        help="Overlap between consecutive chunks"
    )
 
    top_k = st.slider(
        "Top K Results",
        1,
        10,
        3,
        1,
        help="Number of relevant chunks to retrieve"
    )
 
    st.divider()
 
    st.markdown("**Model Information**")
 
    st.caption("🔍 Embeddings: all-MiniLM-L6-v2 (Local)")
    st.caption("🤖 LLM: llama-3.1-8b-instant (Groq)")
    st.caption("☁️ Inference: Groq Free Tier")
    st.caption("🔑 API Key: Enter in sidebar above")
 
    st.divider()
 
    show_chunks = st.toggle(
        "Show Retrieved Chunks",
        value=False
    )
 
 
# -------------------------------------------------------
# HEADER
# -------------------------------------------------------
 
st.title("📚 AI Study Buddy")
 
st.markdown(
    """
Upload a PDF document and ask questions based on its content.
 
This application uses:
- FAISS for Retrieval
- Sentence Transformers for Embeddings
- Groq Free Inference for Answer Generation (LLaMA3)
"""
)
 
st.divider()
 
# -------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------
 
if "chunks" not in st.session_state:
    st.session_state.chunks = None
 
if "index" not in st.session_state:
    st.session_state.index = None
 
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
 
 
# -------------------------------------------------------
# PDF UPLOAD SECTION
# -------------------------------------------------------
 
st.subheader("1️⃣ Upload PDF")
 
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type=["pdf"]
)
 
 
# -------------------------------------------------------
# PDF PROCESSING
# -------------------------------------------------------
 
if uploaded_file is not None:
 
    new_upload = (
        st.session_state.pdf_name
        != uploaded_file.name
    )
 
    if new_upload:
 
        with st.spinner("📖 Processing PDF..."):
 
            try:
 
                raw_text = extract_text(uploaded_file)
 
                if not raw_text:
                    st.error("❌ No readable text found in the PDF.")
                    st.stop()
 
                chunks = create_chunks(
                    raw_text,
                    chunk_size=chunk_size,
                    overlap=overlap
                )
 
                index = create_vector_store(chunks)
 
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.pdf_name = uploaded_file.name
                st.session_state.chat_history = []
 
                st.success(
                    f"✅ {uploaded_file.name} loaded successfully "
                    f"({len(chunks)} chunks | {len(raw_text):,} characters)"
                )
 
            except Exception as e:
                st.error(f"❌ Error: {e}")
                st.stop()
 
    else:
        st.info(
            f"📌 {uploaded_file.name} already loaded "
            f"({len(st.session_state.chunks)} chunks)"
        )
 
 
# -------------------------------------------------------
# CHAT SECTION
# -------------------------------------------------------
 
if st.session_state.chunks is not None:
 
    st.divider()
 
    st.subheader("2️⃣ Ask Questions")
 
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
 
    user_question = st.chat_input(
        "Ask anything about the uploaded PDF..."
    )
 
    if user_question:
 
        # Block if no API key
        if not groq_api_key:
            st.error("❌ Please enter your Groq API key in the sidebar first.")
            st.stop()
 
        with st.chat_message("user"):
            st.markdown(user_question)
 
        st.session_state.chat_history.append(
            {"role": "user", "content": user_question}
        )
 
        results = retrieve_answer(
            question=user_question,
            chunks=st.session_state.chunks,
            index=st.session_state.index,
            top_k=top_k
        )
 
        if show_chunks and results:
            with st.expander("🔍 Retrieved Context Chunks", expanded=False):
                for i, r in enumerate(results, 1):
                    st.markdown(
                        f'<div class="chunk-box">'
                        f'<span class="score-badge">#{i} Score: {r["score"]:.3f}</span><br><br>'
                        f'{r["text"]}</div>',
                        unsafe_allow_html=True
                    )
 
        context = "\n\n".join([r["text"] for r in results])
 
        with st.chat_message("assistant"):
            with st.spinner("🤔 Generating Answer..."):
                answer = ask_groq(context, user_question, groq_api_key)
            st.markdown(answer)
 
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer}
        )
 
    if st.session_state.chat_history:
 
        st.divider()
 
        col1, col2 = st.columns([3, 1])
 
        with col2:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
 
else:
 
    st.divider()
 
    st.info("👆 Please upload a PDF first.")
 
    with st.expander("ℹ️ How Does It Work?"):
        st.markdown("""
### Free RAG Pipeline with Groq
 
| Step | Description |
|------|-------------|
| 🔑 API Key | Enter free Groq key in sidebar |
| 📤 Upload | User uploads a PDF |
| 📖 Extract | Text is extracted from the PDF |
| ✂️ Chunk | Text is split into smaller chunks |
| 🔢 Embed | Embeddings generated using all-MiniLM-L6-v2 |
| 💾 Index | Stored in FAISS vector database |
| ❓ Question | User asks a question |
| 🔍 Search | Relevant chunks are retrieved |
| 🤖 Answer | LLaMA3 via Groq generates the answer |
 
**Get your free Groq API key:** [console.groq.com](https://console.groq.com)
""")
 
st.divider()
 
st.caption(
    "Built with ❤️ using Streamlit · FAISS · SentenceTransformers · Groq Free Inference"
)
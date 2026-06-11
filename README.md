# 📚 AI Study Buddy

An AI-powered PDF Question Answering application built using **Streamlit, FAISS, Sentence Transformers, and Groq LLM**.

Upload any PDF document and ask questions about its content. The application uses a Retrieval-Augmented Generation (RAG) pipeline to find relevant information from the document and generate accurate answers.

---

## 🚀 Features

✅ Upload PDF documents

✅ Extract text from PDFs

✅ Automatic text chunking

✅ Semantic search using embeddings

✅ FAISS vector database

✅ Groq LLaMA 3.1 for answer generation

✅ Chat-style interface

✅ Adjustable chunk size and overlap

✅ View retrieved context chunks

---

## 🏗️ Project Architecture

```text
User
 │
 ▼
Upload PDF
 │
 ▼
PDF Text Extraction
 │
 ▼
Text Chunking
 │
 ▼
Sentence Embeddings
(all-MiniLM-L6-v2)
 │
 ▼
FAISS Vector Store
 │
 ▼
User Question
 │
 ▼
Similarity Search
 │
 ▼
Top Relevant Chunks
 │
 ▼
Groq LLaMA 3.1
 │
 ▼
Generated Answer
```

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|----------|
| Streamlit | Web Application |
| PyPDF | PDF Text Extraction |
| Sentence Transformers | Text Embeddings |
| FAISS | Vector Database |
| NumPy | Numerical Operations |
| Groq API | LLM Inference |
| LLaMA 3.1 8B Instant | Answer Generation |

---

## 📂 Project Structure

```text
AI-Study-Buddy/
│
├── app.py
├── rag_core.py
├── requirements.txt
├── README.md
└── assets/
```

---

## ⚙️ Installation

### Clone Repository

```bash
git clone https://github.com/ShraddhaPatel1906/AI-Study-Buddy.git

cd AI-Study-Buddy
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

Linux / Mac:

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Groq API Key

Get a free API key from:

https://console.groq.com

Enter the key in the application's sidebar.

---

## ▶️ Run Application

```bash
streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## 🧠 How the RAG Pipeline Works

### Step 1: PDF Upload

User uploads a PDF document.

### Step 2: Text Extraction

PyPDF extracts text from each page.

### Step 3: Chunk Creation

Large text is divided into smaller overlapping chunks.

Example:

```text
Chunk 1 → Characters 1–500
Chunk 2 → Characters 401–900
Chunk 3 → Characters 801–1300
```

### Step 4: Embedding Generation

Each chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### Step 5: Vector Storage

Vectors are stored inside FAISS.

### Step 6: User Question

Question is converted into an embedding.

### Step 7: Similarity Search

FAISS retrieves the most relevant chunks.

### Step 8: Answer Generation

Retrieved chunks are sent to Groq LLaMA 3.1.

The model generates an answer using only the retrieved context.

---

## 📸 Screenshots

### Home Page

Add screenshot here

### PDF Upload

Add screenshot here

### Question Answering

Add screenshot here

---

## 🎯 Example Questions

```text
Explain RNN in detail.

What is backpropagation?

What are CNN layers?

Define gradient descent.

Explain transformers.
```

---

## 📈 Future Improvements

- Multiple PDF support
- Chat history export
- Source citations
- Hybrid search
- Local LLM support
- Multi-language PDFs
- Voice input

---

## 👩‍💻 Author

**Shraddha Patel**

M.Sc. Data Science  
IIIT Lucknow

GitHub:

https://github.com/ShraddhaPatel1906

---

## ⭐ If you like this project

Give the repository a star ⭐

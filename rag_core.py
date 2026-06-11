from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re


# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------
# SentenceTransformer is a pre-trained AI model that
# converts text into numerical vectors (embeddings).
#
# "all-MiniLM-L6-v2" is a lightweight but powerful model:
#   - Produces 384-dimensional vectors
#   - Fast and works well even on CPU
#   - Quite accurate for semantic similarity tasks
#
# We load it once at the module level so that it does not
# reload on every function call, which improves performance.
model = SentenceTransformer("all-MiniLM-L6-v2")


# -------------------------------------------------------
# STEP 1: EXTRACT TEXT FROM PDF
# -------------------------------------------------------
def extract_text(pdf_file) -> str:
    """
    Extracts plain text from a PDF file.

    How it works:
        - PdfReader reads the PDF page by page
        - Text is extracted from each page
        - Text from all pages is combined into a single string

    Args:
        pdf_file: PDF file path (string) or file-like object
                  (such as Streamlit's UploadedFile)

    Returns:
        str: Complete text content of the PDF

    Raises:
        Exception: If the PDF is corrupted or cannot be read

    Example:
        text = extract_text("document.pdf")
        print(text[:200])  # View first 200 characters
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            # Some pages may be blank or text extraction may fail,
            # so check whether any text was returned.
            if page_text:
                text += page_text + "\n"  # Add newline between pages

        return text.strip()  # Remove leading/trailing spaces and newlines

    except Exception as e:
        raise Exception(f"PDF reading error: {e}")


# -------------------------------------------------------
# STEP 2: SPLIT TEXT INTO CHUNKS
# -------------------------------------------------------
def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Splits large text into smaller overlapping chunks.

    Why chunking is important:
        - AI models cannot efficiently process very large text at once
        - Searching through smaller chunks is more accurate
        - Vector similarity works better on shorter text segments

    Why use overlap?
        - If an important sentence falls at a chunk boundary,
          overlap ensures that the complete sentence appears
          in at least one chunk
        - Maintains context between neighboring chunks

    How it works (sliding window):

        [========chunk 1========]
                    [========chunk 2========]
                                [========chunk 3========]
        |--chunk_size--|
                    |overlap|

    Args:
        text       : Input text (usually extracted from PDF)
        chunk_size : Maximum length of each chunk in characters. Default: 500
        overlap    : Number of overlapping characters between chunks. Default: 100

    Returns:
        list[str]: List of text chunks

    Example:
        chunks = create_chunks(text, chunk_size=500, overlap=100)
        print(f"Total chunks: {len(chunks)}")
    """
    if not text:
        return []

    # Replace multiple spaces, tabs, and newlines
    # with a single space for cleaner text.
    text = re.sub(r"\s+", " ", text)

    chunks = []
    start = 0  # Starting position of the sliding window

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the window forward while preserving overlap
        # Example:
        # chunk_size=500, overlap=100
        # next start = current_start + 400
        start += (chunk_size - overlap)

    return chunks


# -------------------------------------------------------
# STEP 3: CREATE VECTOR STORE (FAISS DATABASE)
# -------------------------------------------------------
def create_vector_store(chunks: list[str]):
    """
    Converts text chunks into vector embeddings and stores
    them in a FAISS index.

    What are Vector Embeddings?
        - Each chunk is converted into a numerical vector
        - Texts with similar meanings have vectors that are
          close to each other in vector space
        - This is the foundation of semantic search

    What is FAISS (Facebook AI Similarity Search)?
        - An open-source library developed by Meta (Facebook)
        - Can find nearest vectors among millions in milliseconds
        - IndexFlatIP uses Inner Product similarity
          (equivalent to cosine similarity when vectors are normalized)

    Why normalization?
        - normalize_embeddings=True converts vectors to unit length
        - Inner Product then becomes Cosine Similarity
        - Similarity range: [-1, +1]
            +1 = identical meaning
             0 = unrelated
            -1 = opposite meaning

    Args:
        chunks: List of text chunks generated by create_chunks()

    Returns:
        faiss.Index: Populated FAISS index object

    Raises:
        ValueError: If chunks list is empty

    Example:
        index = create_vector_store(chunks)
        print(f"Vectors stored: {index.ntotal}")
    """
    if len(chunks) == 0:
        raise ValueError("No chunks found — no text available to process")

    # Generate embeddings using Sentence Transformer
    # Shape: (num_chunks, 384)
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    dimension = embeddings.shape[1]  # 384 for MiniLM

    # Create FAISS index
    # IndexFlatIP = exact search using Inner Product
    # Approximate methods such as IndexIVFFlat are used
    # for very large datasets.
    index = faiss.IndexFlatIP(dimension)

    # Convert embeddings to float32 and add to FAISS
    # because FAISS expects float32 input.
    index.add(embeddings.astype("float32"))

    return index


# -------------------------------------------------------
# STEP 4: RETRIEVE RELEVANT CHUNKS FOR A QUESTION
# -------------------------------------------------------
def retrieve_answer(
    question: str,
    chunks: list[str],
    index,
    top_k: int = 3
) -> list[dict]:
    """
    Finds and returns the most relevant text chunks
    for a user's question.

    How it works:
        1. Convert the question into a vector using the same model
        2. Search FAISS for the top_k nearest vectors
        3. Return the corresponding original text chunks

    This is semantic search:
    It matches based on meaning rather than exact words.

    Example:
        Question: "What was the company's revenue?"
        Chunk:    "Total earnings for the fiscal year were 50 crore."
        -> This can match because "revenue" and "earnings"
           have similar meanings.

    Args:
        question : User question (string)
        chunks   : Original list of text chunks
        index    : FAISS index created by create_vector_store()
        top_k    : Number of best matching chunks to return. Default: 3

    Returns:
        list[dict]: Each dictionary contains:
            - "score" (float): Similarity score (0 to 1, higher is better)
            - "text"  (str): Original chunk text

    Example:
        results = retrieve_answer("Who is the CEO?", chunks, index, top_k=3)

        for r in results:
            print(f"Score: {r['score']:.2f} | Text: {r['text'][:100]}")
    """
    # Convert question into a vector
    # [question] is used because model.encode expects a batch
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # Search in FAISS
    # Returns:
    # scores  -> similarity values
    # indices -> positions of matching chunks
    scores, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        # Safety check:
        # FAISS may return -1 if insufficient results are found
        if idx < len(chunks):
            results.append(
                {
                    "score": float(score),  # NumPy float -> Python float
                    "text": chunks[idx]
                }
            )

    return results
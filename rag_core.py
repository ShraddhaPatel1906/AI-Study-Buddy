from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re
 
 
# -------------------------------------------------------
# MODEL LOADING
# -------------------------------------------------------
# SentenceTransformer ek pre-trained AI model hai jo
# text ko numbers (vectors/embeddings) mein convert karta hai.
#
# "all-MiniLM-L6-v2" ek lightweight lekin powerful model hai:
#   - 384-dimension vectors produce karta hai
#   - Fast hai aur CPU par bhi achhe se chalta hai
#   - Semantic similarity ke liye kaafi accurate hai
#
# Isko module level par ek baar load karte hain taaki
# har function call par dobara load na ho — performance ke liye zaroori.
model = SentenceTransformer("all-MiniLM-L6-v2")
 
 
# -------------------------------------------------------
# STEP 1: PDF SE TEXT NIKALNA
# -------------------------------------------------------
def extract_text(pdf_file) -> str:
    """
    PDF file se plain text extract karta hai.
 
    Kaise kaam karta hai:
        - PdfReader PDF ko page-by-page padhta hai
        - Har page ka text extract hota hai
        - Sab pages ka text ek saath jod diya jata hai
 
    Args:
        pdf_file: PDF file ka path (string) ya file-like object
                  (jaise Streamlit ka UploadedFile)
 
    Returns:
        str: PDF ka saara text ek string mein
 
    Raises:
        Exception: Agar PDF corrupt ho ya read na ho sake
    
    Example:
        text = extract_text("document.pdf")
        print(text[:200])  # pehle 200 characters dekho
    """
    try:
        reader = PdfReader(pdf_file)
        text = ""
 
        for page in reader.pages:
            page_text = page.extract_text()
 
            # Kuch pages blank hote hain ya extract nahi hota,
            # isliye check karte hain ki kuch mila bhi ya nahi
            if page_text:
                text += page_text + "\n"  # pages ke beech newline
 
        return text.strip()  # aage-peeche ke spaces/newlines hatao
 
    except Exception as e:
        raise Exception(f"PDF reading error: {e}")
 
 
# -------------------------------------------------------
# STEP 2: TEXT KO CHUNKS MEIN TODNA
# -------------------------------------------------------
def create_chunks(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    Bade text ko chhote overlapping tukdon (chunks) mein todta hai.
 
    Chunking kyun zaroori hai?
        - AI models ek baar mein bahut zyada text process nahi kar sakte
        - Chhote chunks mein search karna zyada accurate hota hai
        - Vector similarity short texts mein better kaam karta hai
 
    Overlap kyun rakhte hain?
        - Agar koi important sentence chunk ki boundary par ho
          toh overlap ensure karta hai ki woh kisi na kisi chunk mein poora mile
        - Context maintain rehta hai chunks ke beech
 
    Kaise kaam karta hai (sliding window):
 
        [========chunk 1========]
                    [========chunk 2========]
                                [========chunk 3========]
        |--chunk_size--|
                    |overlap|
 
    Args:
        text       : Input text (PDF se nikala hua)
        chunk_size : Har chunk ki maximum length (characters mein). Default: 500
        overlap    : Consecutive chunks ke beech overlap (characters mein). Default: 100
 
    Returns:
        list[str]: Text chunks ki list
 
    Example:
        chunks = create_chunks(text, chunk_size=500, overlap=100)
        print(f"Total chunks: {len(chunks)}")
    """
    if not text:
        return []
 
    # Multiple spaces, tabs, newlines ko single space se replace karo
    # Taaki text clean aur consistent rahe
    text = re.sub(r"\s+", " ", text)
 
    chunks = []
    start = 0  # sliding window ka starting position
 
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]  # is window ka text nikaalo
        chunks.append(chunk)
 
        # Agla chunk overlap ke baad shuru hoga
        # Agar overlap=100 aur chunk_size=500 toh:
        # next start = current_start + 400
        start += (chunk_size - overlap)
 
    return chunks
 
 
# -------------------------------------------------------
# STEP 3: VECTOR STORE BANANA (FAISS DATABASE)
# -------------------------------------------------------
def create_vector_store(chunks: list[str]):
    """
    Text chunks ko vector embeddings mein convert karke
    FAISS index mein store karta hai.
 
    Vector Embeddings kya hote hain?
        - Har chunk ko ek list of numbers (vector) mein convert kiya jata hai
        - Similar meaning wale texts ke vectors ek doosre ke paas hote hain
        - Yahi "semantic search" ka aadhaar hai
 
    FAISS (Facebook AI Similarity Search) kya hai?
        - Meta (Facebook) ka open-source library hai
        - Lakho vectors mein se bhi milliseconds mein nearest neighbors dhundh sakta hai
        - IndexFlatIP = Inner Product similarity use karta hai
          (normalized vectors ke saath yeh cosine similarity ke barabar hai)
 
    Normalization kyun?
        - normalize_embeddings=True karne se sab vectors unit length ke ho jaate hain
        - Inner product tab cosine similarity ban jaata hai: [-1, +1] range
        - +1 = bilkul same meaning, 0 = unrelated, -1 = opposite meaning
 
    Args:
        chunks: Text chunks ki list (create_chunks() se milti hai)
 
    Returns:
        faiss.Index: Populated FAISS index object
 
    Raises:
        ValueError: Agar chunks list empty ho
 
    Example:
        index = create_vector_store(chunks)
        print(f"Vectors stored: {index.ntotal}")
    """
    if len(chunks) == 0:
        raise ValueError("No chunks found — koi text nahi mila process karne ke liye")
 
    # Sentence Transformer se embeddings generate karo
    # Shape: (num_chunks, 384) — 384 dimensions per chunk
    embeddings = model.encode(
        chunks,
        convert_to_numpy=True,       # PyTorch tensor ki jagah NumPy array chahiye FAISS ke liye
        normalize_embeddings=True    # Unit length normalization (cosine similarity ke liye)
    )
 
    dimension = embeddings.shape[1]  # = 384 for MiniLM model
 
    # FAISS Index banao — IndexFlatIP = exact search with Inner Product
    # (Approximate methods jaise IndexIVFFlat bade datasets ke liye use hote hain)
    index = faiss.IndexFlatIP(dimension)
 
    # Embeddings ko float32 mein convert karke index mein daalo
    # FAISS float32 expect karta hai
    index.add(embeddings.astype("float32"))
 
    return index
 
 
# -------------------------------------------------------
# STEP 4: QUESTION KE LIYE RELEVANT CHUNKS DHUNDHNA
# -------------------------------------------------------
def retrieve_answer(
    question: str,
    chunks: list[str],
    index,
    top_k: int = 3
) -> list[dict]:
    """
    User ke question ke liye sabse relevant text chunks dhundh ke return karta hai.
 
    Kaise kaam karta hai:
        1. Question ko bhi vector mein convert karo (same model se)
        2. FAISS mein is question vector ke sabse paas ke top_k vectors dhundho
        3. Un vectors ke corresponding original text chunks return karo
 
    Yeh "semantic search" hai — exact word matching nahi,
    balki meaning ki similarity dekhi jaati hai.
 
    Misal:
        Question: "company ka revenue kitna tha?"
        Chunk:    "fiscal year mein total earnings 50 crore rahi"
        → Yeh match hoga kyunki "revenue" aur "earnings" ka meaning similar hai
 
    Args:
        question : User ka sawal (string)
        chunks   : Original text chunks ki list
        index    : FAISS index (create_vector_store() se)
        top_k    : Kitne best matching chunks chahiye. Default: 3
 
    Returns:
        list[dict]: Har dict mein do cheezein hain:
            - "score" (float): Similarity score, 0 to 1 (1 = perfect match)
            - "text"  (str):   Original chunk text
 
    Example:
        results = retrieve_answer("CEO kaun hai?", chunks, index, top_k=3)
        for r in results:
            print(f"Score: {r['score']:.2f} | Text: {r['text'][:100]}")
    """
    # Question ko vector mein convert karo
    # [question] — list isliye kyunki model.encode batch expect karta hai
    question_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
 
    # FAISS mein search karo
    # Returns: scores (similarity values), indices (chunk positions)
    scores, indices = index.search(
        question_embedding.astype("float32"),
        top_k
    )
 
    results = []
    for score, idx in zip(scores[0], indices[0]):
        # Safety check: kabhi kabhi FAISS -1 return karta hai
        # agar enough results nahi milte
        if idx < len(chunks):
            results.append(
                {
                    "score": float(score),  # NumPy float → Python float
                    "text": chunks[idx]     # Index se original chunk
                }
            )
 
    return results
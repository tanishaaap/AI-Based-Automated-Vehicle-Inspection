import os
import re
from sentence_transformers import SentenceTransformer
import chromadb

# --- Load the embedding model (downloads once, ~80MB, then cached locally) ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# --- Set up Chroma — this creates a local folder-based database, no server needed ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../rag_llm  (ingest.py's own folder)
DEFAULT_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "documents")  # .../rag_llm/documents

VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")
client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection("vehicle_inspection_kb")

# --- Path fix: resolve relative to THIS FILE's location, not the current working directory ---
# ingest.py lives at: <project_root>/backend/rag/ingest.py
# knowledge/sources lives at: <project_root>/knowledge/sources
# So we go up 2 levels from this script's folder to reach project_root.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # .../rag_llm  (ingest.py's own folder)
DEFAULT_KNOWLEDGE_DIR = os.path.join(BASE_DIR, "documents")  # .../rag_llm/documents

def clean_text(text):
    """Strip citation-export artifacts like [cite: 5] or [cite: 1, 5] before chunking.
    These are leftover markup from wherever the doc was authored/exported and add
    noise to the embeddings without adding meaning."""
    text = re.sub(r"\[cite:\s*[\d,\s]+\]", "", text)
    return text


def chunk_text(text, chunk_size=300, overlap=50):
    """Split text into overlapping word chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap  # overlap keeps context from being cut mid-idea
    return chunks


def ingest_folder(folder_path=None):
    if folder_path is None:
        folder_path = DEFAULT_KNOWLEDGE_DIR

    if not os.path.isdir(folder_path):
        raise FileNotFoundError(
            f"Knowledge folder not found at: {folder_path}\n"
            f"Create it and place your .txt reference files there, or pass the correct "
            f"path explicitly: ingest_folder(r'C:\\path\\to\\your\\knowledge\\sources')"
        )

    for filename in os.listdir(folder_path):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
            text = f.read()

        text = clean_text(text)
        chunks = chunk_text(text)
        embeddings = embedder.encode(chunks).tolist()

        ids = [f"{filename}_{i}" for i in range(len(chunks))]
        metadatas = [{"source": filename} for _ in chunks]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        print(f"Ingested {filename}: {len(chunks)} chunks")


if __name__ == "__main__":
    print(f"Looking for reference docs in: {DEFAULT_KNOWLEDGE_DIR}")
    ingest_folder()
    print("Done. Total chunks in DB:", collection.count())
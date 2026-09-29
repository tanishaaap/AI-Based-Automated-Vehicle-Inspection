import os
from sentence_transformers import SentenceTransformer
import chromadb

embedder = SentenceTransformer("all-MiniLM-L6-v2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

client = chromadb.PersistentClient(path=VECTOR_DB_DIR)
collection = client.get_or_create_collection("vehicle_inspection_kb")

def retrieve(query, top_k=3):
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    return results["documents"][0]  # list of the top_k matching text chunks

if __name__ == "__main__":
    test_query = "is a 4mm panel gap acceptable?"
    for chunk in retrieve(test_query):
        print("---")
        print(chunk)
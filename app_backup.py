from fastapi import FastAPI
from pydantic import BaseModel
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=query.question
    )
    return {"answer": response.text}


def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

notes_text = read_text("sample_notes.txt")
print(notes_text[:500])  # just to check



import nltk
nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def chunk_text(text, max_words=50):  # smaller chunks for demo
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""
    current_len = 0

    for sentence in sentences:
        words = sentence.split()
        if current_len + len(words) <= max_words:
            current_chunk += sentence + " "
            current_len += len(words)
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
            current_len = len(words)

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

chunks = chunk_text(notes_text)
print(f"Total chunks: {len(chunks)}")
print(chunks)


from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Load a small embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for all chunks
chunk_embeddings = model.encode(chunks)

# Convert to float32 (FAISS requirement)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

# Create FAISS index
dimension = chunk_embeddings.shape[1]  # embedding size
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

print("FAISS index created with", index.ntotal, "chunks")


def get_relevant_chunks(question, top_k=3):
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results
def get_relevant_chunks(question, top_k=3):
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results

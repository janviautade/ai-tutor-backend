# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# --- Step 2.2: Load fake lesson notes ---
def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()

notes_text = read_text("sample_notes.txt")
print("First 200 chars of notes:")
print(notes_text[:200])

# --- Step 2.3: Split into chunks (simple hack, no NLTK) ---
def chunk_text(text):
    # Split every line and sentence separately
    chunks = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences = line.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence + ".")
    return chunks


chunks = chunk_text(notes_text)
print(f"Total chunks: {len(chunks)}")
print(chunks)

# --- Step 3.2: Create embeddings and FAISS index ---
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print("FAISS index created with", index.ntotal, "chunks")

# --- Step 3.3: Function to search relevant chunks ---
def get_relevant_chunks(question, top_k=1):
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
    return results


# --- Step 3.4: FastAPI endpoint ---
app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Add this right after app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_ai(query: Query):
    relevant_chunks = get_relevant_chunks(query.question)
    if relevant_chunks:
        # Turn the chunk into a human-readable answer
        answer = f"Here's what I found about '{query.question}': {relevant_chunks[0]}"
    else:
        answer = "Sorry, I don't have info on that topic yet."
    return {"answer": answer, "sources": relevant_chunks}


# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import os

# ------------------------------
# Configure Gemini API key
# ------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY", "AIzaSyBOya1n1rqvfsGkJTSawO1C4CQsnsgDC-Q"))

# ------------------------------
# FastAPI app and CORS
# ------------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------------------
# Load all notes from sample_notes folder
# ------------------------------
def read_all_texts(folder_path):
    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                texts.append(f.read())
    return "\n".join(texts)

notes_text = read_all_texts("sample_notes")
print("First 200 chars of notes:")
print(notes_text[:200])

# ------------------------------
# Split text into chunks
# ------------------------------
def chunk_text(text):
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

# ------------------------------
# Create embeddings and FAISS index
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)
print("FAISS index created with", index.ntotal, "chunks")

# ------------------------------
# Function to get top relevant chunks
# ------------------------------
def get_relevant_chunks(question, top_k=6):  # increased to 6 chunks
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    results = [chunks[i] for i in indices[0]]  # take top_k chunks without distance filtering
    return results

# ------------------------------
# FastAPI request model
# ------------------------------
class Query(BaseModel):
    question: str

# ------------------------------
# Ask endpoint: FAISS + Gemini
# ------------------------------
@app.post("/ask")
def ask_ai(query: Query):
    relevant_chunks = get_relevant_chunks(query.question, top_k=6)
    print("Relevant chunks:", relevant_chunks)
    context = "\n".join(relevant_chunks) if relevant_chunks else ""

    if context:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Answer the question in 4–5 sentences using the following notes:\n{context}\n\nQuestion: {query.question}"
        )
        answer = response.text
    else:
        answer = "Sorry, I don't have info on that topic yet."

    return {"answer": answer, "sources": relevant_chunks}

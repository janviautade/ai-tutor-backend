# app.py
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from fastapi.middleware.cors import CORSMiddleware
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Configure Gemini API key
# ------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

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
# Test route to verify backend is running
# ------------------------------
@app.get("/")
def home():
    return {"status": "Backend running successfully ✅"}

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

# ------------------------------
# Create embeddings and FAISS index
# ------------------------------
model = SentenceTransformer('all-MiniLM-L6-v2')
chunk_embeddings = model.encode(chunks)
chunk_embeddings = np.array(chunk_embeddings).astype("float32")

dimension = chunk_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(chunk_embeddings)

# ------------------------------
# Function to get top relevant chunks
# ------------------------------
def get_relevant_chunks(question, top_k=6):
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    results = [chunks[i] for i in indices[0]]
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
    context = "\n".join(relevant_chunks) if relevant_chunks else ""

    if context:
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Answer the question in 4–5 sentences using the following notes:\n{context}\n\nQuestion: {query.question}"
            )
            answer = response.text or "Sorry, I don't have enough information to answer that question."
        except Exception as e:
            print("Gemini API error:", e)
            answer = "Sorry, I don't have enough information to answer that question."
    else:
        answer = "Sorry, I don't have enough information to answer that question."

    # Return only the answer (no sources)
    return {"answer": answer}

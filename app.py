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
import gc

load_dotenv()

# ------------------------------
# Configure Gemini API key
# ------------------------------
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# ------------------------------
# FastAPI app and CORS
# ------------------------------
app = FastAPI(title="AI Tutor API", description="Memory-optimized for Railway deployment")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for lazy loading - initialized only when needed
model = None
chunks = None
index = None
notes_text = None

# ------------------------------
# Load all notes from sample_notes folder (lazy)
# ------------------------------
def read_all_texts(folder_path):
    """Load text only when needed"""
    if notes_text is not None:
        return notes_text

    texts = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    texts.append(content)
    return "\n".join(texts)

# ------------------------------
# Split text into chunks (lazy)
# ------------------------------
def get_chunks():
    """Get chunks only when needed"""
    global chunks, notes_text
    if chunks is not None:
        return chunks

    notes_text = read_all_texts("sample_notes")
    chunks = []
    for line in notes_text.splitlines():
        line = line.strip()
        if not line:
            continue
        sentences = line.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence + ".")
    return chunks

# ------------------------------
# Initialize model and embeddings (lazy loading)
# ------------------------------
def initialize_model():
    """Initialize model only when needed"""
    global model
    if model is None:
        model = SentenceTransformer('all-MiniLM-L6-v2')

def initialize_embeddings():
    """Initialize embeddings and FAISS index only when needed"""
    global chunks, index
    if index is not None:
        return

    chunks = get_chunks()
    initialize_model()

    # Process in smaller batches to reduce memory usage during initialization
    batch_size = 20
    all_embeddings = []

    for i in range(0, len(chunks), batch_size):
        batch_chunks = chunks[i:i + batch_size]
        batch_embeddings = model.encode(batch_chunks)
        all_embeddings.extend(batch_embeddings)

        # Cleanup memory after each batch
        gc.collect()

    chunk_embeddings = np.array(all_embeddings).astype("float32")
    dimension = chunk_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(chunk_embeddings)

# ------------------------------
# Function to get top relevant chunks
# ------------------------------
def get_relevant_chunks(question, top_k=3):  # Reduced from 6 to 3 for memory efficiency
    """Get relevant chunks with lazy loading"""
    initialize_embeddings()

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
# Health check endpoint
# ------------------------------
@app.get("/health")
def health_check():
    """Simple health check"""
    return {"status": "healthy", "chunks_loaded": chunks is not None}

# ------------------------------
# Ask endpoint: FAISS + Gemini (optimized)
# ------------------------------
@app.post("/ask")
def ask_ai(query: Query):
    """Answer questions with memory optimization"""
    try:
        if not query.question.strip():
            return {"answer": "Please provide a question.", "sources": []}

        relevant_chunks = get_relevant_chunks(query.question, top_k=3)
        context = "\n".join(relevant_chunks) if relevant_chunks else ""

        if context:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=f"Answer the question in 3-4 sentences using the following notes:\n{context}\n\nQuestion: {query.question}"
            )
            answer = response.text if response.text else "I couldn't generate an answer."
        else:
            answer = "Sorry, I don't have enough information to answer that question."

        # Cleanup memory after processing
        gc.collect()

        return {"answer": answer, "sources": relevant_chunks}

    except Exception as e:
        # Cleanup on error
        gc.collect()
        return {"answer": "An error occurred while processing your question.", "sources": []}

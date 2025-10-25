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
# Load all notes from sample_notes folder
# ------------------------------
def read_all_texts(folder_path):
    texts = []
    lesson_counter = 1
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            with open(os.path.join(folder_path, file_name), "r", encoding="utf-8") as f:
                text = f.read()
                texts.append({"lesson": f"Lesson {lesson_counter}", "text": text})
                lesson_counter += 1
    return texts

lessons = read_all_texts("sample_notes")

# ------------------------------
# Split text into chunks and track lesson sources
# ------------------------------
chunks = []
chunk_sources = []
for lesson in lessons:
    lines = lesson["text"].splitlines()
    for line in lines:
        line = line.strip()
        if not line:
            continue
        sentences = line.split(".")
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                chunks.append(sentence + ".")
                chunk_sources.append(lesson["lesson"])

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
def get_relevant_chunks(question, top_k=6, similarity_threshold=0.55):
    q_embedding = model.encode([question]).astype("float32")
    distances, indices = index.search(q_embedding, top_k)
    
    # convert L2 distance to cosine similarity approximation
    similarities = 1 / (1 + distances[0])
    
    results = []
    sources = []
    for idx, sim in zip(indices[0], similarities):
        if sim >= similarity_threshold:
            results.append(chunks[idx])
            sources.append(chunk_sources[idx])
    
    return results, sources

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
    question = query.question.strip()
    
    # quick check for gibberish / empty question
    if not question or all(not c.isalpha() for c in question):
        return {"answer": "Sorry, I don't have enough information to answer that question.", "sources": []}

    relevant_chunks, relevant_sources = get_relevant_chunks(question, top_k=6)

    # Only keep meaningful chunks (more than 1 word)
    meaningful_chunks = []
    meaningful_sources = []
    for c, s in zip(relevant_chunks, relevant_sources):
        if c.strip() and len(c.split()) > 1:
            meaningful_chunks.append(c)
            meaningful_sources.append(s)

    if not meaningful_chunks:
        # Out-of-domain / gibberish → fallback immediately
        return {"answer": "Sorry, I don't have enough information to answer that question.", "sources": []}

    context = "\n".join(meaningful_chunks)

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=f"Answer the question in 4–5 sentences using the following notes:\n{context}\n\nQuestion: {question}"
        )
        answer = response.text or "Sorry, I don't have enough information to answer that question."
    except Exception:
        answer = "Sorry, I don't have enough information to answer that question."

    # return answer with lesson sources
    return {"answer": answer, "sources": list(set(meaningful_sources))}

# ------------------------------
# Simple health check endpoint
# ------------------------------
@app.get("/")
def home():
    return {"status": "Backend running successfully ✅"}

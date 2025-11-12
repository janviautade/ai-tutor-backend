from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
import json
import time
from dotenv import load_dotenv

load_dotenv()

# ------------------------------
# Try to load Gemini API first (most important for AI responses)
# ------------------------------
gemini_loaded = False
client = None
try:
    from google import genai
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
    gemini_loaded = True
    print("Gemini API loaded successfully")
except Exception as e:
    print(f"Gemini API unavailable: {e}")
    print("AI responses will use fallback mode")

# ------------------------------
# Try to load ML dependencies for embeddings
# ------------------------------
ml_loaded = False
vector_db_loaded = False
try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
    import faiss
    ml_loaded = True
    vector_db_loaded = True
    print("ML dependencies and FAISS loaded successfully")
except ImportError as e:
    print(f"FAISS not available: {e}")
    # Try fallback vector search without FAISS
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        ml_loaded = True
        print("ML dependencies loaded, using sklearn for vector search")
    except ImportError as e2:
        print(f"ML dependencies not available: {e2}")
        print("Using keyword-based content matching")
except Exception as e:
    print(f"Error loading ML dependencies: {e}")
    print("Using keyword-based content matching")

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
# Database setup
# ------------------------------
Base = declarative_base()

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)

class ChatMessage(Base):
    __tablename__ = 'chat_messages'
    id = Column(Integer, primary_key=True)
    text = Column(Text)
    from_bot = Column(Integer)  # 0 for user, 1 for bot
    sources = Column(Text)  # JSON string of sources
    timestamp = Column(Integer)  # Unix timestamp

class Feedback(Base):
    __tablename__ = 'feedback'
    id = Column(Integer, primary_key=True)
    message_id = Column(Integer)  # References ChatMessage.id
    helpful = Column(Integer)  # 1 for helpful, 0 for not helpful
    timestamp = Column(Integer)  # Unix timestamp

engine = create_engine('sqlite:///lessons.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

# ------------------------------
# Load all notes from sample_notes folder (for initial population)
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

# Populate DB if empty
session = Session()
if session.query(Lesson).count() == 0:
    texts = read_all_texts("sample_notes")
    for text in texts:
        lesson = Lesson(title=text['lesson'], content=text['text'])
        session.add(lesson)
    session.commit()
session.close()

# ------------------------------
# Load all notes from database
# ------------------------------
def load_lessons_from_db():
    session = Session()
    lessons = session.query(Lesson).all()
    session.close()
    return [{'lesson': l.title, 'text': l.content} for l in lessons]

lessons = load_lessons_from_db()

# ------------------------------
# Always create text chunks for fallback functionality
# ------------------------------
chunks = []
chunk_sources = []

# Split text into chunks and track lesson sources (always available)
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

print(f"Loaded {len(chunks)} text chunks from {len(lessons)} lessons")

# ------------------------------
# Initialize ML components if available
# ------------------------------
model = None
index = None

if ml_loaded:
    # Create embeddings and FAISS index
    try:
        model = SentenceTransformer('all-MiniLM-L6-v2')
        chunk_embeddings = model.encode(chunks)
        chunk_embeddings = np.array(chunk_embeddings).astype("float32")

        dimension = chunk_embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(chunk_embeddings)
        print("FAISS index created successfully")
    except Exception as e:
        print(f"Error creating FAISS index: {e}")
        ml_loaded = False
else:
    print("Skipping ML initialization - running in limited mode")

# ------------------------------
# Function to get top relevant chunks using vector search
# ------------------------------
def get_relevant_chunks(question, top_k=6, similarity_threshold=0.3):
    if not ml_loaded or model is None:
        # Fallback: simple keyword matching when ML is not available
        question_lower = question.lower()
        results = []
        sources = []

        for chunk, source in zip(chunks, chunk_sources):
            chunk_lower = chunk.lower()
            # Simple relevance check: if question keywords appear in chunk
            keywords = [word for word in question_lower.split() if len(word) > 3]
            if any(keyword in chunk_lower for keyword in keywords):
                results.append(chunk)
                sources.append(source)
                if len(results) >= top_k:
                    break

        return results[:top_k], sources[:top_k]

    # Use vector search (FAISS or sklearn)
    q_embedding = model.encode([question]).astype("float32")

    if vector_db_loaded and index is not None:
        # Use FAISS for vector search
        distances, indices = index.search(q_embedding, top_k)
        # Convert L2 distance to similarity score (higher is better)
        similarities = 1 / (1 + distances[0])

        results = []
        sources = []
        for idx, sim in zip(indices[0], similarities):
            if sim >= similarity_threshold:
                results.append(chunks[idx])
                sources.append(chunk_sources[idx])

        return results, sources

    else:
        # Use sklearn cosine similarity as fallback
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            # Get embeddings for all chunks (create if not exists)
            if not hasattr(get_relevant_chunks, 'chunk_embeddings'):
                get_relevant_chunks.chunk_embeddings = model.encode(chunks).astype("float32")

            # Calculate cosine similarities
            similarities = cosine_similarity(q_embedding, get_relevant_chunks.chunk_embeddings)[0]

            # Get top similar chunks
            top_indices = similarities.argsort()[-top_k:][::-1]  # Sort descending

            results = []
            sources = []
            for idx in top_indices:
                if similarities[idx] >= similarity_threshold:
                    results.append(chunks[idx])
                    sources.append(chunk_sources[idx])

            return results, sources

        except Exception as e:
            print(f"Vector search failed, falling back to keyword matching: {e}")
            # Final fallback to keyword matching
            question_lower = question.lower()
            results = []
            sources = []

            for chunk, source in zip(chunks, chunk_sources):
                chunk_lower = chunk.lower()
                keywords = [word for word in question_lower.split() if len(word) > 3]
                if any(keyword in chunk_lower for keyword in keywords):
                    results.append(chunk)
                    sources.append(source)
                    if len(results) >= top_k:
                        break

            return results[:top_k], sources[:top_k]

# ------------------------------
# FastAPI request model
# ------------------------------
class Query(BaseModel):
    question: str

def extract_key_info_from_content(context, question):
    """Extract and format key information from lesson content when AI is unavailable"""
    # Split context into sentences
    sentences = [s.strip() for s in context.split('.') if s.strip()]

    # Look for sentences that contain question keywords
    question_lower = question.lower()
    keywords = [word for word in question_lower.split() if len(word) > 2]

    relevant_sentences = []
    for sentence in sentences:
        sentence_lower = sentence.lower()
        if any(keyword in sentence_lower for keyword in keywords):
            relevant_sentences.append(sentence)

    if relevant_sentences:
        # Return just the key information without extra formatting
        if len(relevant_sentences) == 1:
            return relevant_sentences[0].strip()
        else:
            # Combine multiple relevant sentences
            return " ".join(relevant_sentences[:2]).strip()
    else:
        # If no specific matches, provide a general summary
        return context[:200].strip()

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

    # Use Gemini to summarize the relevant content (content-only, no external knowledge)
    if gemini_loaded and client is not None:
        try:
            prompt = f"Answer the question using only the following lesson content. Do not add any external knowledge or information not present in the provided content. Provide a detailed answer with at least 5-6 sentences explaining the topic clearly and comprehensively:\n\nContent: {context}\n\nQuestion: {question}\n\nProvide a comprehensive answer based only on the content above."
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            if response.text and response.text.strip():
                answer = response.text.strip()
            else:
                # Fallback to content summary if Gemini returns empty
                answer = extract_key_info_from_content(context, question)
        except Exception as e:
            print(f"Gemini summarization failed: {e}")
            # Fallback: extract key information from content
            answer = extract_key_info_from_content(context, question)
    else:
        # Fallback when Gemini not available
        answer = extract_key_info_from_content(context, question)

    # return answer with lesson sources
    return {"answer": answer, "sources": list(set(meaningful_sources))}

# ------------------------------
# Resources endpoint: Get available lesson resources
# ------------------------------
@app.get("/resources")
def get_resources():
    session = Session()
    lessons = session.query(Lesson).all()
    session.close()

    resources = []
    for lesson in lessons:
        # Map lesson to resource format expected by frontend
        # Assume files are named lesson{id}.pdf
        resources.append({
            "name": lesson.title,
            "file": f"lesson{lesson.id}.pdf"
        })

    return {"resources": resources}

# ------------------------------
# Progress endpoint: Get class progress data
# ------------------------------
@app.get("/progress")
def get_progress():
    # Mock progress data - in a real app this would come from a students/progress table
    progress = [
        {"student": "Alice", "completed": 3, "total": 5},
        {"student": "Bob", "completed": 4, "total": 5},
        {"student": "Charlie", "completed": 2, "total": 5},
        {"student": "Diana", "completed": 5, "total": 5}
    ]
    return {"progress": progress}

# ------------------------------
# Chat history endpoints
# ------------------------------
class ChatMessageRequest(BaseModel):
    text: str
    from_bot: int  # 0 for user, 1 for bot
    sources: list = []  # Optional list of sources

@app.get("/chat/history")
def get_chat_history():
    session = Session()

    # Get messages with their feedback status
    messages_query = session.query(ChatMessage).order_by(ChatMessage.timestamp).all()

    # Get all feedback for bot messages
    feedback_data = {}
    if messages_query:
        bot_message_ids = [msg.id for msg in messages_query if msg.from_bot]
        if bot_message_ids:
            feedback_query = session.query(Feedback).filter(Feedback.message_id.in_(bot_message_ids)).all()
            for fb in feedback_query:
                feedback_data[fb.message_id] = {
                    "helpful": bool(fb.helpful),
                    "feedbackGiven": True
                }

    session.close()

    history = []
    for msg in messages_query:
        sources = json.loads(msg.sources) if msg.sources else []
        message_data = {
            "id": msg.id,
            "text": msg.text,
            "fromBot": bool(msg.from_bot),
            "sources": sources,
            "timestamp": msg.timestamp
        }

        # Add feedback status for bot messages
        if msg.from_bot and msg.id in feedback_data:
            message_data.update(feedback_data[msg.id])

        history.append(message_data)

    return {"messages": history}

@app.post("/chat/message")
def save_chat_message(message: ChatMessageRequest):
    session = Session()
    chat_msg = ChatMessage(
        text=message.text,
        from_bot=message.from_bot,
        sources=json.dumps(message.sources),
        timestamp=int(time.time())
    )
    session.add(chat_msg)
    session.commit()

    # Get the ID and timestamp before closing the session
    msg_id = chat_msg.id
    msg_timestamp = chat_msg.timestamp

    session.close()

    return {"id": msg_id, "timestamp": msg_timestamp}

@app.delete("/chat/history")
def clear_chat_history():
    session = Session()
    session.query(ChatMessage).delete()
    session.commit()
    session.close()

    return {"message": "Chat history cleared"}

# ------------------------------
# Feedback endpoints
# ------------------------------
class FeedbackRequest(BaseModel):
    message_id: int
    helpful: int  # 1 for helpful, 0 for not helpful

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    session = Session()
    fb = Feedback(
        message_id=feedback.message_id,
        helpful=feedback.helpful,
        timestamp=int(time.time())
    )
    session.add(fb)
    session.commit()
    session.close()

    return {"message": "Feedback submitted successfully"}

@app.get("/feedback/analytics")
def get_feedback_analytics():
    session = Session()

    # Get feedback stats by lesson/source - count each feedback once per message
    feedback_data = session.query(
        Feedback.helpful,
        ChatMessage.sources,
        Feedback.id  # Include feedback ID to avoid duplicates
    ).join(ChatMessage, Feedback.message_id == ChatMessage.id).all()

    session.close()

    # Process feedback data - each feedback counts once, but distributed across sources
    lesson_feedback = {}
    total_feedback = {"helpful": 0, "not_helpful": 0, "total": 0}
    processed_feedbacks = set()  # Track processed feedback IDs

    for helpful, sources_str, feedback_id in feedback_data:
        if feedback_id in processed_feedbacks:
            continue  # Skip if already processed this feedback

        processed_feedbacks.add(feedback_id)
        sources = json.loads(sources_str) if sources_str else []

        # Count this feedback once in total
        if helpful == 1:
            total_feedback["helpful"] += 1
        else:
            total_feedback["not_helpful"] += 1
        total_feedback["total"] += 1

        # Distribute across sources (if message has multiple sources, each gets credit)
        if sources:
            for source in sources:
                if source not in lesson_feedback:
                    lesson_feedback[source] = {"helpful": 0, "not_helpful": 0, "total": 0}

                if helpful == 1:
                    lesson_feedback[source]["helpful"] += 1
                else:
                    lesson_feedback[source]["not_helpful"] += 1

                lesson_feedback[source]["total"] += 1
        else:
            # If no sources, put in "Unknown" category
            if "Unknown" not in lesson_feedback:
                lesson_feedback["Unknown"] = {"helpful": 0, "not_helpful": 0, "total": 0}

            if helpful == 1:
                lesson_feedback["Unknown"]["helpful"] += 1
            else:
                lesson_feedback["Unknown"]["not_helpful"] += 1

            lesson_feedback["Unknown"]["total"] += 1

    return {
        "lesson_feedback": lesson_feedback,
        "total_feedback": total_feedback
    }

# ------------------------------
# Simple health check endpoint
# ------------------------------
@app.get("/")
def home():
    return {"status": "Backend running successfully ✅"}

from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os

Base = declarative_base()

class Lesson(Base):
    __tablename__ = 'lessons'
    id = Column(Integer, primary_key=True)
    title = Column(String)
    content = Column(Text)

engine = create_engine('sqlite:///lessons.db')
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)

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
    print("Database populated with lessons.")
else:
    print("Database already has lessons.")

# Test loading
lessons = session.query(Lesson).all()
print(f"Loaded {len(lessons)} lessons from database.")
for lesson in lessons:
    print(f"- {lesson.title}: {lesson.content[:50]}...")

session.close()

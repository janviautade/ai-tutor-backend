import google.generativeai as genai

genai.configure(api_key="AIzaSyBOya1n1rqvfsGkJTSawO1C4CQsnsgDC-Q")

response = genai.models.generate(
    model="chat-bison-001",
    prompt="Hello AI, can you answer this?"
)

print(response)






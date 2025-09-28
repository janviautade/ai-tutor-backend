import google.generativeai as genai

genai.configure(api_key="AIzaSyBOya1n1rqvfsGkJTSawO1C4CQsnsgDC-Q")

# Try with a safe model
model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("Hello Gemini! Explain the solar system in simple words.")
print(response.text)

import React, { useState } from "react";
import axios from "axios";

function Chat() {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState([]);
  const [history, setHistory] = useState([]);

  const askQuestion = async () => {
    if (!question.trim()) return;
    try {
      const res = await axios.post("http://127.0.0.1:8000/ask", { question });
      setAnswer(res.data.answer);
      setSources(res.data.sources);
      setHistory([{ question, answer: res.data.answer }, ...history]);
      setQuestion("");
    } catch (err) {
      console.error(err);
      setAnswer("Error: Could not fetch answer.");
      setSources([]);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "600px", margin: "0 auto" }}>
      <h2>AI Tutor Chat</h2>
      <input
        type="text"
        value={question}
        onChange={(e) => setQuestion(e.target.value)}
        placeholder="Ask a question..."
        style={{ width: "80%", padding: "8px" }}
      />
      <button onClick={askQuestion} style={{ padding: "8px 12px", marginLeft: "8px" }}>
        Ask
      </button>

      {answer && (
  <div style={{ marginTop: "20px", padding: "10px", border: "1px solid #ccc" }}>
    <p><strong>Answer:</strong> {answer}</p>
    {sources.length > 0 && (
      <div>
        <strong>Sources:</strong>
        <ul>
          {sources.map((s, idx) => (
            <li key={idx}>{s}</li>
          ))}
        </ul>
      </div>
    )}
    <div style={{ marginTop: "10px" }}>
      <span>Was this helpful? </span>
      <button
        onClick={() => alert("Thanks for your feedback! 👍")}
        style={{ marginRight: "5px" }}
      >
        Yes
      </button>
      <button onClick={() => alert("Thanks! We’ll improve. 👎")}>No</button>
    </div>
  </div>
)}


      {history.length > 0 && (
        <div style={{ marginTop: "30px" }}>
          <h3>Previous Questions</h3>
          <ul>
            {history.map((h, idx) => (
              <li key={idx}>
                <strong>Q:</strong> {h.question} <br />
                <strong>A:</strong> {h.answer}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

export default Chat;

import os
import requests
import gradio as gr
from dotenv import load_dotenv
from pdf_loader import load_pdf_chunks
from rag_agent import MathRAGAgent

# Load environment variables
load_dotenv()

# Initialize agent
pdf_path = "numerical_methods.pdf"
agent = MathRAGAgent()

if not os.path.exists(agent.index_file):
    print("🔨 Building embeddings from PDF…")
    chunks = load_pdf_chunks(pdf_path)
    agent.create_embeddings(chunks)
else:
    print("📂 Loading existing embeddings…")
    agent.load_embeddings()

# OpenRouter chat helper
def query_openrouter(prompt: str,
                     model: str = "mistralai/mistral-7b-instruct") -> str:
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a helpful math tutor. Explain clearly with steps."},
            {"role": "user", "content": prompt}
        ]
    }
    resp = requests.post("https://openrouter.ai/api/v1/chat/completions",
                         headers=headers, json=data)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# Gradio callback
def answer_math_question(user_query: str) -> str:
    contexts = agent.search(user_query, top_k=3)
    context_text = "\n\n".join(contexts)
    full_prompt = (
        f"Answer the question using this context:\n\n{context_text}\n\n"
        f"Question: {user_query}"
    )
    return query_openrouter(full_prompt)

# Build UI
demo = gr.Interface(
    fn=answer_math_question,
    inputs=gr.Textbox(lines=2, placeholder="Ask your numerical methods question…"),
    outputs="text",
    title="Numerical Methods Tutor",
    description="Upload your PDF, then ask any question. Powered by RAG + OpenRouter."
)

if __name__ == "__main__":
    demo.launch()

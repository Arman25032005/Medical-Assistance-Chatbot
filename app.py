import os
import logging
from flask import Flask, render_template, jsonify, request
from dotenv import load_dotenv
from src.helper import get_embeddings
from src.prompt import *  # expects system_prompt
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not PINECONE_API_KEY or not GEMINI_API_KEY:
    raise RuntimeError("Missing PINECONE_API_KEY or GEMINI_API_KEY")

embedding = get_embeddings()
INDEX_NAME = "medical-assistant-chatbot"

vector_store = PineconeVectorStore.from_existing_index(
    index_name=INDEX_NAME,
    embedding=embedding,
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

qa_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, qa_chain)

@app.route("/")
def index():
    return render_template("chat.html")

# (Optional legacy endpoint retained)
@app.route("/get", methods=["POST"])
def chat_legacy():
    msg = request.form.get("msg", "").strip()
    if not msg:
        return "Empty message", 400
    try:
        result = rag_chain.invoke({"input": msg})
        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = "I am unable to find an answer in the current knowledge base."
        return answer
    except Exception:
        logger.exception("Error in /get")
        return "Server error", 500

@app.route("/api/chat", methods=["POST"])
def chat_api():
    data = request.get_json(silent=True) or {}
    msg = (data.get("message") or "").strip()
    if not msg:
        return jsonify({"error": "Empty message"}), 400
    try:
        result = rag_chain.invoke({"input": msg})
        answer = (result.get("answer") or "").strip()
        if not answer:
            answer = "I am unable to find an answer in the current knowledge base."
        return jsonify({"answer": answer})
    except Exception:
        logger.exception("Error in /api/chat")
        return jsonify({"error": "Server error"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
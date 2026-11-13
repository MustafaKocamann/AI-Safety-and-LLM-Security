"""
A basic RAG application built with LangGraph.

This file contains the same logic demonstrated in rag.ipynb.
The Streamlit app (app.py) and security notebooks import from here,
so the "baseline app" lives in one place.

Flow:   question -> retrieve -> generate -> answer
"""

import os
from typing import TypedDict

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from fake_docs import DOCUMENTS

load_dotenv()

# .env configurations
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
TOP_K = 2

SYSTEM_PROMPT = (
    "You are the Novacure help assistant. "
    "Answer the user's question using only the context below. "
    "If the answer is not in the context, say \"I don't know based on the company documents.\" "
    "Keep answers short."
)


# ---------------------------------------------------------------------------
# 1. The graph state: the data that flows between the steps
# ---------------------------------------------------------------------------
class RAGState(TypedDict):
    question: str
    context: list[Document]
    answer: str


# ---------------------------------------------------------------------------
# 2. Knowledge base: fake documents -> embeddings -> vector store
# ---------------------------------------------------------------------------
def load_documents() -> list[Document]:
    return [
        Document(page_content=d["text"], metadata={"id": d["id"], "title": d["title"]})
        for d in DOCUMENTS
    ]


def build_vector_store(embeddings=None) -> InMemoryVectorStore:
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    store = InMemoryVectorStore(embeddings)
    store.add_documents(load_documents())
    return store


# ---------------------------------------------------------------------------
# 3. Prompt building: system prompt + retrieved passages + user message
# ---------------------------------------------------------------------------
def build_messages(question: str, context: list[Document]) -> list[tuple[str, str]]:
    passages = "\n\n".join(f"[{d.metadata['title']}]\n{d.page_content}" for d in context)
    return [
        ("system", f"{SYSTEM_PROMPT}\n\nContext:\n{passages}"),
        ("user", question),
    ]


# ---------------------------------------------------------------------------
# 4. The graph: retrieve node -> generate node
# ---------------------------------------------------------------------------
def build_rag_graph(llm=None, embeddings=None, vector_store=None):
    if llm is None:
        llm = ChatOpenAI(
            model=CHAT_MODEL,
            temperature=0,
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
        )
    if vector_store is None:
        vector_store = build_vector_store(embeddings)

    def retrieve(state: RAGState) -> dict:
        docs = vector_store.similarity_search(state["question"], k=TOP_K)
        return {"context": docs}

    def generate(state: RAGState) -> dict:
        messages = build_messages(state["question"], state["context"])
        response = llm.invoke(messages)
        return {"answer": response.content}

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


def ask(app, question: str) -> RAGState:
    return app.invoke({"question": question})

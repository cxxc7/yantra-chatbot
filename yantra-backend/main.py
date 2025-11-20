import os
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import chromadb
import cohere
from groq import Groq

# ==========================
# ENVIRONMENT
# ==========================

load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not COHERE_API_KEY:
    raise RuntimeError("Missing COHERE_API_KEY in .env")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in .env")

co = cohere.Client(COHERE_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================
# FASTAPI
# ==========================

app = FastAPI(
    title="YantraLive RAG Chatbot (Cohere + Groq)",
    version="1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # OK for dev, tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# CHROMA (no embedding_function; we pass embeddings manually)
# ==========================

chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="yantra_end_customer")

# ==========================
# Pydantic MODELS
# ==========================

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

class ChatResponse(BaseModel):
    answer: str
    used_context: List[str]
    from_fallback: bool = False

# ==========================
# COHERE EMBEDDINGS
# ==========================

def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Use Cohere to embed dataset rows as 'search_document'.
    """
    resp = co.embed(
        texts=texts,
        model="embed-english-v3.0",
        input_type="search_document",
    )
    # resp.embeddings is a list of vectors
    return resp.embeddings

def embed_query(text: str) -> List[float]:
    """
    Use Cohere to embed user query as 'search_query'.
    """
    resp = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_query",
    )
    return resp.embeddings[0]

# ==========================
# LOAD CSV + INDEX (Cohere embeddings -> Chroma)
# ==========================

DATA_FILE = "data/end_customer.csv"

def load_and_index():
    if not os.path.exists(DATA_FILE):
        raise RuntimeError(f"Dataset file not found at: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)
    if df.empty:
        raise RuntimeError("Dataset is empty")

    documents: List[str] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        # Turn each row into a single text "col: value | col: value | ..."
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(row_text)
        ids.append(f"row_{i}")

    try:
        vectors = embed_documents(documents)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
        )
        print(f"Indexed {len(documents)} rows using Cohere embeddings.")
    except Exception as e:
        # Do not crash the server if embedding fails; start with empty index
        print(f"[WARN] Failed to embed/index dataset with Cohere: {e}")
        print("[WARN] Starting server without vector index; chat will fallback.")

load_and_index()

# ==========================
# FALLBACK MESSAGE
# ==========================

def fallback() -> str:
    return (
        "I couldn't find this information in the latest YantraLive dataset.\n\n"
        f"Please contact human support:\n"
        f"📞 {SUPPORT_PHONE}\n"
        f"📧 {SUPPORT_EMAIL}"
    )

# ==========================
# GROQ GENERATION (Llama 3.3)
# ==========================

GROQ_MODEL_ID = "llama-3.3-70b-versatile"

def generate_with_groq(context: str, user_question: str) -> Optional[str]:
    """
    Use Groq's Llama 3.3 model to answer STRICTLY from context.
    If answer not clearly in context, it should reply UNSURE_FROM_DATA.
    """
    system_prompt = (
        "You are a strict RAG assistant for YantraLive END-CUSTOMER rock breaker data.\n"
        "\n"
        "GENERAL RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT.\n"
        "- If the answer is not clearly present in the CONTEXT, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT guess. Do NOT use outside knowledge.\n"
        "- Answer clearly and in a structured way for the end customer.\n"
        "\n"
        "FOR COMPATIBILITY QUESTIONS (e.g., 'which breaker is compatible with SANY SY20', "
        "'which all breakers work with Hyundai R30', 'show options for X machine'):\n"
        "- Scan ALL lines in the CONTEXT.\n"
        "- Identify EVERY row where the machine brand and/or machine model match the user question.\n"
        "- Collect all DISTINCT compatible breaker models / SKUs from those rows.\n"
        "- Return them as a BULLET LIST, one breaker per line.\n"
        "- For each breaker, include key details if present: breaker model name, SKU, chisel diameter, "
        "impact energy, and any important notes (like price or stock).\n"
        "- Do NOT arbitrarily pick only one breaker if multiple are present; always show all relevant options.\n"
        "\n"
        "FOR SPECIFIC PARAMETER QUESTIONS (e.g., 'What is the chisel diameter for Hyundai R30?', "
        "'What is the impact energy in joules for model VJ20 HD?'):\n"
        "- Find the row(s) that match the machine/breaker mentioned.\n"
        "- Extract the exact requested numeric or textual value from those row(s).\n"
        "- If multiple rows give different values, mention each distinct value clearly.\n"
        "\n"
        "REMEMBER:\n"
        "- Never invent breakers or values that are not present in the CONTEXT.\n"
        "- If the machine or breaker mentioned is not present in the CONTEXT at all, reply UNSURE_FROM_DATA.\n"
    )

    user_content = f"""
CONTEXT (rows from YantraLive END-CUSTOMER dataset):
{context}

USER QUESTION:
{user_question}
"""

    try:
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
        )
        answer = resp.choices[0].message.content
        return answer
    except Exception as e:
        print(f"[GROQ ERROR] {e}")
        return None

# ==========================
# ROUTES
# ==========================

@app.get("/api/health")
def health():
    return {"status": "ok"}

@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    # Last user message
    user_msg = req.messages[-1].content
    print(f"[CHAT] User message: {user_msg!r}")

    # 1) Embed the query with Cohere
    try:
        query_vec = embed_query(user_msg)
    except Exception as e:
        print(f"[ERROR] Failed to embed user query with Cohere: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    # 2) Retrieve similar rows from Chroma
    try:
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=25,
        )
        docs = result["documents"][0] if result["documents"] else []
    except Exception as e:
        print(f"[ERROR] Failed to query Chroma: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    if not docs:
        print("[CHAT] No relevant documents found in index.")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    context = "\n\n---\n\n".join(docs)

    # 3) Ask Groq to answer using this context
    raw = generate_with_groq(context=context, user_question=user_msg)
    if raw is None:
        # Groq failed → human fallback
        return ChatResponse(
            answer=fallback(),
            used_context=docs,
            from_fallback=True,
        )

    raw = raw.strip()

    # 4) If Groq says UNSURE_FROM_DATA → human fallback
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=docs,
            from_fallback=True,
        )

    # 5) Normal answer
    return ChatResponse(
        answer=raw,
        used_context=docs,
        from_fallback=False,
    )

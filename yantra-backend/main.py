import os
from typing import List, Literal, Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import chromadb
from openai import OpenAI
from openai import APIError

# ==========================
# ENVIRONMENT
# ==========================

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================

app = FastAPI(
    title="YantraLive RAG Chatbot (OpenAI)",
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
# OPENAI EMBEDDINGS
# ==========================

def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Use OpenAI to embed a list of texts.
    """
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    # resp.data is a list of objects with .embedding
    vectors: List[List[float]] = [d.embedding for d in resp.data]
    return vectors

# ==========================
# LOAD CSV + INDEX
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
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        documents.append(row_text)
        ids.append(f"row_{i}")

    try:
        vectors = embed_text(documents)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=vectors,
        )
        print(f"Indexed {len(documents)} rows using OpenAI embeddings.")
    except Exception as e:
        # Do not crash the server if embedding fails; start with empty index
        print(f"[WARN] Failed to embed/index dataset: {e}")
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
# OPENAI GENERATION (tries multiple models)
# ==========================

MODEL_CANDIDATES = [
    "gpt-4o-mini",  # fast, cheap
    "gpt-4.1",      # more accurate
]

def generate_with_openai(prompt: str) -> Optional[str]:
    """
    Try several OpenAI chat models in order.
    If one fails, try the next.
    If all fail, return None.
    """
    last_err: Optional[Exception] = None

    for model_id in MODEL_CANDIDATES:
        try:
            print(f"[OPENAI] Trying model: {model_id}")
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a strict RAG assistant for YantraLive "
                            "end-customer rock breaker data. "
                            "Use ONLY the given context. "
                            "If the answer is not clearly present, reply EXACTLY: UNSURE_FROM_DATA."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.0,
            )
            raw = resp.choices[0].message.content
            if raw:
                print(f"[OPENAI] Success with model: {model_id}")
                return raw
            else:
                print(f"[OPENAI] Model {model_id} returned empty content.")
        except APIError as e:
            print(f"[OPENAI] APIError from model {model_id}: {e}")
            last_err = e
            continue
        except Exception as e:
            print(f"[OPENAI] Error from model {model_id}: {e}")
            last_err = e
            # for non-API errors, break to avoid spamming
            break

    print(f"[OPENAI] All candidate models failed. Last error: {last_err}")
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

    # 1) Embed the query
    try:
        query_vec = embed_text([user_msg])[0]
    except Exception as e:
        print(f"[ERROR] Failed to embed user query: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    # 2) Retrieve similar rows from Chroma
    try:
        result = collection.query(
            query_embeddings=[query_vec],
            n_results=5,
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

    # 3) Build prompt content (user side; system is in generate_with_openai)
    prompt = f"""
CONTEXT (rows from YantraLive end-customer dataset):
{context}

USER QUESTION:
{user_msg}
"""

    # 4) Call OpenAI via helper (trying multiple model IDs)
    raw = generate_with_openai(prompt)
    if raw is None:
        # All model attempts failed -> human fallback
        return ChatResponse(
            answer=fallback(),
            used_context=docs,
            from_fallback=True,
        )

    raw = raw.strip()

    # 5) Handle unsure case -> human fallback
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=docs,
            from_fallback=True,
        )

    # 6) Normal answer case
    return ChatResponse(
        answer=raw,
        used_context=docs,
        from_fallback=False,
    )

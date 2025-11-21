import os
import time
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
    version="1.1",
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

# Separate collections for each dataset type
end_customer_collection = chroma_client.create_collection(
    name="yantra_end_customer"
)
spare_parts_collection = chroma_client.create_collection(
    name="yantra_spare_parts"
)
dealer_collection = chroma_client.create_collection(
    name="yantra_dealers"
)

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
# COHERE EMBEDDINGS + RETRY
# ==========================

EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 64  # good for ~450 row CSVs
EMBED_BATCH_SLEEP_SECONDS = 1.0  # pause between batches to avoid token-per-minute issues


def _cohere_embed_with_retry(
    texts: List[str],
    input_type: str,
    label: str = "",
    max_retries: int = 5,
) -> List[List[float]]:
    """
    Generic Cohere embed helper with simple retry/backoff on rate limits.
    """
    for attempt in range(max_retries):
        try:
            resp = co.embed(
                texts=texts,
                model=EMBED_MODEL,
                input_type=input_type,
            )
            return resp.embeddings
        except Exception as e:
            msg = str(e).lower()
            # very simple detection of rate-limit style errors
            if "rate limit" in msg or "429" in msg:
                wait = 5 * (attempt + 1)
                print(
                    f"[COHERE] Rate limited while embedding {label} "
                    f"(attempt {attempt + 1}/{max_retries}). Sleeping {wait}s."
                )
                time.sleep(wait)
                continue

            print(f"[COHERE] Non-rate-limit error while embedding {label}: {e}")
            raise

    raise RuntimeError(f"Cohere embed retries exceeded for {label}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    """
    Use Cohere to embed dataset rows as 'search_document'.
    (Caller is responsible for batching to respect limits.)
    """
    return _cohere_embed_with_retry(
        texts=texts,
        input_type="search_document",
        label=f"documents batch (size={len(texts)})",
    )


def embed_query(text: str) -> List[float]:
    """
    Use Cohere to embed user query as 'search_query'.
    """
    embeddings = _cohere_embed_with_retry(
        texts=[text],
        input_type="search_query",
        label="user query",
    )
    return embeddings[0]


# ==========================
# LOAD CSV + INDEX (Cohere embeddings -> Chroma)
# ==========================

DATA_DIR = "data"
END_CUSTOMER_FILE = os.path.join(DATA_DIR, "end_customer.csv")
SPARE_PARTS_FILE = os.path.join(DATA_DIR, "spare_parts.csv")
DEALERS_FILE = os.path.join(DATA_DIR, "dealers.csv")


def load_and_index_one(path: str, collection, tag: str):
    """
    Generic loader: reads a CSV and indexes it into a given Chroma collection.
    Each row becomes one text, prefixed with [TAG] to indicate dataset type.
    Batches embedding calls to respect Cohere trial limits.
    """
    if not os.path.exists(path):
        print(f"[INFO] Dataset not found for {tag}: {path} (skipping)")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Dataset {tag} is empty: {path}")
        return

    documents: List[str] = []
    ids: List[str] = []

    # NOTE: if you want to reduce token usage even more,
    # you can restrict to a subset of important columns here.
    for i, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
        # Tag so the model knows what kind of data it is
        documents.append(f"[{tag}] {row_text}")
        ids.append(f"{tag.lower()}_row_{i}")

    total_docs = len(documents)
    print(f"[INDEX] Starting indexing for {tag}: {total_docs} rows")

    try:
        for start in range(0, total_docs, EMBED_BATCH_SIZE):
            end = min(start + EMBED_BATCH_SIZE, total_docs)
            batch_docs = documents[start:end]
            batch_ids = ids[start:end]

            try:
                batch_vectors = embed_documents(batch_docs)
            except Exception as batch_err:
                print(
                    f"[WARN] Failed to embed batch {start}:{end} for {tag}: {batch_err}"
                )
                # Skip this batch but continue with others
                continue

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_vectors,
            )
            print(f"[INDEX] Indexed rows {start} to {end - 1} for {tag}")

            # small pause to avoid hitting token-per-minute limits
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

        print(
            f"[INDEX] Finished indexing {total_docs} rows for {tag} using Cohere embeddings."
        )
    except Exception as e:
        print(f"[WARN] Failed to embed/index dataset {tag} with Cohere: {e}")
        print("[WARN] Starting server without this vector index; chat may fallback.")


def load_all_datasets():
    load_and_index_one(END_CUSTOMER_FILE, end_customer_collection, "END_CUSTOMER")
    # small pause between full datasets
    time.sleep(5)

    load_and_index_one(SPARE_PARTS_FILE, spare_parts_collection, "SPARE_PARTS")
    time.sleep(5)

    load_and_index_one(DEALERS_FILE, dealer_collection, "DEALERS")


# Run indexing on startup
load_all_datasets()

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
        "You are a strict RAG assistant for YantraLive END-CUSTOMER, SPARE PARTS, "
        "and DEALER rock breaker data.\n"
        "\n"
        "GENERAL RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT.\n"
        "- If the answer is not clearly present in the CONTEXT, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT guess. Do NOT use outside knowledge.\n"
        "- Keep answers concise, factual, and formatted cleanly.\n"
        "- Respect the dataset tags [END_CUSTOMER], [SPARE_PARTS], [DEALERS] when reasoning.\n"
        "\n"
        "===================================================\n"
        "COMPATIBILITY QUESTIONS\n"
        "(e.g., 'Which breaker is compatible with SANY SY20?',\n"
        "'Which all breakers work with Hyundai R30?', 'Show options for X machine')\n"
        "===================================================\n"
        "- Scan EVERY row in the CONTEXT.\n"
        "- Identify all rows where the machine brand and/or machine model match the user query.\n"
        "- Extract all DISTINCT compatible breaker models / SKUs.\n"
        "- Answer TO-THE-POINT using this exact format:\n"
        "\n"
        "**Compatible Breakers:**\n"
        "- <Breaker Model / SKU> – <key facts from dataset only>\n"
        "- <Breaker Model / SKU> – <key facts from dataset only>\n"
        "\n"
        "- Do NOT add explanations, stories, or filler text.\n"
        "- Do NOT pick only one breaker; list ALL valid options.\n"
        "- Include only factual attributes found in the CONTEXT (e.g., chisel diameter, impact energy,\n"
        "  recommended tonnage, price, stock, or important notes) when they are present.\n"
        "\n"
        "===================================================\n"
        "COMPARISON QUESTIONS\n"
        "(e.g., 'Compare JCB and CAT', 'Compare breaker A vs breaker B')\n"
        "===================================================\n"
        "- Start your answer with: 'Here is a comparison between <X> and <Y>:'\n"
        "- Build a clean Markdown table using this format:\n"
        "  | Feature | Option 1 | Option 2 |\n"
        "  |--------|----------|----------|\n"
        "  | ...    | ...      | ...      |\n"
        "- Choose meaningful features from the CONTEXT such as machine model, breaker model,\n"
        "  impact energy, chisel diameter, price, stock, etc.\n"
        "- Only include facts that are clearly present in the CONTEXT.\n"
        "\n"
        "===================================================\n"
        "SPECIFIC PARAMETER QUESTIONS\n"
        "(e.g., 'What is the chisel diameter for Hyundai R30?',\n"
        "'What is the impact energy in joules for model VJ20 HD?')\n"
        "===================================================\n"
        "- Find the row(s) that match the machine/breaker mentioned.\n"
        "- Extract the exact requested numeric or textual value from those row(s).\n"
        "- If multiple rows give different values, mention each distinct value clearly.\n"
        "- Keep the answer short and directly focused on the requested parameter.\n"
        "\n"
        "===================================================\n"
        "SUBJECTIVE / BEST-OPTION QUESTIONS\n"
        "(e.g., 'Which is the best model for X?', 'Which breaker is more suitable for Y machine?')\n"
        "===================================================\n"
        "- First, list all relevant options as a bullet list with their key specs.\n"
        "- Then, based ONLY on the CONTEXT (features such as impact energy, recommended tonnage,\n"
        "  application type, or any hints in the data), choose ONE option as the best.\n"
        "- Clearly say: 'According to the available data, the best option is <NAME> because ...'\n"
        "- Do NOT invent reasons that are not supported by the CONTEXT.\n"
        "\n"
        "===================================================\n"
        "REMEMBER\n"
        "===================================================\n"
        "- Never invent breakers, dealers, or spare parts that are not present in the CONTEXT.\n"
        "- If the machine, breaker, part, or dealer mentioned is not present at all,\n"
        "  reply UNSURE_FROM_DATA.\n"
    )

    user_content = f"""
CONTEXT (rows from YantraLive datasets):
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

    # 2) Retrieve similar rows from all collections
    docs: List[str] = []

    def _query_collection(coll, label: str):
        try:
            result = coll.query(
                query_embeddings=[query_vec],
                n_results=10,  # enough to not miss alternatives
            )
            return result["documents"][0] if result["documents"] else []
        except Exception as e:
            print(f"[ERROR] Failed to query {label} collection: {e}")
            return []

    docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER"))
    docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS"))
    docs.extend(_query_collection(dealer_collection, "DEALERS"))

    if not docs:
        print("[CHAT] No relevant documents found in any index.")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    # De-duplicate docs while preserving order
    unique_docs = list(dict.fromkeys(docs))
    context = "\n\n---\n\n".join(unique_docs)

    # 3) Ask Groq to answer using this context
    raw = generate_with_groq(context=context, user_question=user_msg)
    if raw is None:
        # Groq failed → human fallback
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    raw = raw.strip()

    # 4) If Groq says UNSURE_FROM_DATA → human fallback
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    # 5) Normal answer
    return ChatResponse(
        answer=raw,
        used_context=unique_docs,
        from_fallback=False,
    )

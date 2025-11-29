import os
import time
import re
from typing import List, Literal, Optional, Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import chromadb
from openai import OpenAI

# ==========================
# ENVIRONMENT
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

# instantiate new-style client (v1.0+)
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI(
    title="YantraLive RAG Chatbot (OpenAI)",
    version="1.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Serve brochures (static) + preview wrapper endpoints
# ==========================
BROCHURE_DIR = os.path.join("data", "brochures")
os.makedirs(BROCHURE_DIR, exist_ok=True)

# mount static (not used for preview wrapper)
app.mount("/static_brochures", StaticFiles(directory=BROCHURE_DIR), name="static_brochures")


def _build_brochure_map():
    m = {}
    if not os.path.isdir(BROCHURE_DIR):
        return m
    for fname in os.listdir(BROCHURE_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        key = re.sub(r"[^a-z0-9]", "", name_no_ext.lower())
        m[key] = fname
    return m


BROCHURE_MAP = _build_brochure_map()


@app.get("/api/brochures/list")
def list_brochures():
    return JSONResponse(BROCHURE_MAP)


@app.get("/brochures/view/{filename}", response_class=HTMLResponse)
def brochure_view(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(BROCHURE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Brochure not found")

    raw_url = f"/brochures/raw/{safe}"

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>{safe} — Brochure</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    html,body {{ height:100%; margin:0; background:#f7f7f7; }}
    .topbar {{ padding:10px; background:#fff; border-bottom:1px solid #eee; display:flex; gap:8px; align-items:center; }}
    .open-btn {{ padding:6px 10px; border-radius:6px; border:1px solid #ccc; background:#fff; text-decoration:none; color:#111; font-size:13px; }}
    .iframe-wrap {{ height: calc(100% - 52px); }}
    iframe {{ width:100%; height:100%; border:none; }}
    .fallback {{ padding:20px; font-family:system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial; color:#222; }}
  </style>
</head>
<body>
  <div class="topbar">
    <strong>{safe}</strong>
    <a class="open-btn" href="{raw_url}" target="_blank" rel="noopener noreferrer">Open in new tab</a>
    <a class="open-btn" href="{raw_url}" download>Download</a>
  </div>
  <div class="iframe-wrap" role="document">
    <iframe src="{raw_url}#toolbar=1" title="brochure">
      <div class="fallback">
        Your browser couldn't display the PDF inline. <a href="{raw_url}" target="_blank" rel="noopener noreferrer">Open brochure</a>
      </div>
    </iframe>
  </div>
</body>
</html>"""
    return HTMLResponse(content=html, status_code=200)


@app.get("/brochures/raw/{filename}")
def brochure_raw(filename: str):
    safe = os.path.basename(filename)
    path = os.path.join(BROCHURE_DIR, safe)
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Brochure not found")

    headers = {
        "Content-Disposition": f'inline; filename="{safe}"',
        "Accept-Ranges": "bytes",
        "X-Content-Type-Options": "nosniff",
        "Cache-Control": "public, max-age=0, must-revalidate",
    }
    return FileResponse(path, media_type="application/pdf", headers=headers)


# ==========================
# CHROMA
# ==========================
chroma_client = chromadb.Client()


def create_or_get_collection(name: str):
    try:
        return chroma_client.create_collection(name=name)
    except Exception:
        return chroma_client.get_collection(name=name)


end_customer_collection = create_or_get_collection("yantra_end_customer")
spare_parts_collection = create_or_get_collection("yantra_spare_parts")
dealer_collection = create_or_get_collection("yantra_dealers")


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
    brochure_url: Optional[str] = None


# ==========================
# OPENAI EMBEDDINGS + RETRY (v1.0+)
# ==========================
EMBED_BATCH_SIZE = 64
EMBED_BATCH_SLEEP_SECONDS = 1.0


def _openai_embed_with_retry(
    texts: List[str],
    model: str,
    label: str = "",
    max_retries: int = 5,
) -> List[List[float]]:
    for attempt in range(max_retries):
        try:
            resp = openai_client.embeddings.create(model=model, input=texts)
            # support both dict-like and object-like responses
            data = getattr(resp, "data", resp.get("data") if isinstance(resp, dict) else None)
            if data is None:
                # try indexing as mapping
                data = resp["data"]
            embeddings = []
            for d in data:
                # d may be Obj or dict
                if isinstance(d, dict):
                    embeddings.append(d.get("embedding"))
                else:
                    # object-like
                    embeddings.append(getattr(d, "embedding", None))
            return embeddings
        except Exception as e:
            msg = str(e).lower()
            if "rate limit" in msg or "429" in msg:
                wait = 5 * (attempt + 1)
                print(
                    f"[OPENAI] Rate limited while embedding {label} "
                    f"(attempt {attempt + 1}/{max_retries}). Sleeping {wait}s."
                )
                time.sleep(wait)
                continue

            print(f"[OPENAI] Non-rate-limit error while embedding {label}: {e}")
            raise

    raise RuntimeError(f"OpenAI embed retries exceeded for {label}")


def embed_documents(texts: List[str]) -> List[List[float]]:
    return _openai_embed_with_retry(
        texts=texts,
        model=OPENAI_EMBED_MODEL,
        label=f"documents batch (size={len(texts)})",
    )


def embed_query(text: str) -> List[float]:
    embeddings = _openai_embed_with_retry(
        texts=[text],
        model=OPENAI_EMBED_MODEL,
        label="user query",
    )
    return embeddings[0]


# ==========================
# LOAD CSV + INDEX (OpenAI -> Chroma)
# ==========================
DATA_DIR = "data"
END_CUSTOMER_FILE = os.path.join(DATA_DIR, "end_customer.csv")
SPARE_PARTS_FILE = os.path.join(DATA_DIR, "spare_parts.csv")
DEALERS_FILE = os.path.join(DATA_DIR, "dealers.csv")


def load_and_index_one(path: str, collection, tag: str):
    if not os.path.exists(path):
        print(f"[INFO] Dataset not found for {tag}: {path} (skipping)")
        return

    df = pd.read_csv(path)
    if df.empty:
        print(f"[WARN] Dataset {tag} is empty: {path}")
        return

    documents: List[str] = []
    ids: List[str] = []

    for i, row in df.iterrows():
        row_text = " | ".join([f"{col}: {row[col]}" for col in df.columns])
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
                continue

            collection.add(
                ids=batch_ids,
                documents=batch_docs,
                embeddings=batch_vectors,
            )
            print(f"[INDEX] Indexed rows {start} to {end - 1} for {tag}")
            time.sleep(EMBED_BATCH_SLEEP_SECONDS)

        print(
            f"[INDEX] Finished indexing {total_docs} rows for {tag} using OpenAI embeddings."
        )
    except Exception as e:
        print(f"[WARN] Failed to embed/index dataset {tag} with OpenAI: {e}")
        print("[WARN] Starting server without this vector index; chat may fallback.")


def load_all_datasets():
    load_and_index_one(END_CUSTOMER_FILE, end_customer_collection, "END_CUSTOMER")
    time.sleep(2)
    load_and_index_one(SPARE_PARTS_FILE, spare_parts_collection, "SPARE_PARTS")
    time.sleep(2)
    load_and_index_one(DEALERS_FILE, dealer_collection, "DEALERS")


try:
    load_all_datasets()
except Exception as e:
    print(f"[WARN] Dataset indexing failed on startup: {e}")


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
# SB -> VJ normalizer (internal only)
# ==========================
def auto_normalize_sb(text: str) -> str:
    if not text:
        return text

    pattern = re.compile(r"\b(SB)(?:[-\s]*)(\d*)\b", re.IGNORECASE)

    def repl(m: re.Match) -> str:
        digits = m.group(2) or ""
        return "VJ" + digits

    return pattern.sub(repl, text)


# ==========================
# GENERATION (OpenAI) with prefix rule
# ==========================
# Keep your system prompt exactly as provided.
GROQ_MODEL_ID = "llama-3.3-70b-versatile"  # kept for parity only

def _extract_choice_message(choice: Any) -> str:
    """
    Robustly extract message content from a choice entry returned by the client.
    Handles dict-like or object-like shapes.
    """
    msg = None
    if isinstance(choice, dict):
        msg = choice.get("message") or choice.get("text") or choice.get("content")
    else:
        msg = getattr(choice, "message", None) or getattr(choice, "text", None) or getattr(choice, "content", None)

    # msg may itself be a dict with 'content' or a string; handle both
    if isinstance(msg, dict):
        # older-style chat choice: {'role': 'assistant', 'content': '...'}
        # or {'content': '...'}
        return msg.get("content") or msg.get("text") or ""
    if isinstance(msg, str):
        return msg
    # object-like message with .content
    return getattr(msg, "content", "") if msg is not None else ""


def generate_with_groq(context: str, user_question: str) -> Optional[str]:
    prefix_rule = (
        "IMPORTANT – ANSWERING GUIDELINES (apply these before any other instruction):\n"
        "- Treat SB-* mentions internally as VJ-* (do this silently). Never mention or explain this mapping to the user.\n"
        "- Do NOT include any provenance or extraction notes (e.g., 'extracted from ...') in the reply; show only the answer.\n"
        "- If the user mentions ONLY a machine model (e.g., 'Hyundai R30') return ALL details present in the CONTEXT for that machine model.\n"
        "  Provide full rows / all dataset columns and keep language natural and helpful.\n"
        "- If the user mentions ONLY a breaker model (e.g., 'VJ20 HD') return ALL details present in the CONTEXT for that breaker model,\n"
        "  EXCLUDING the 'compatible machines' section initially. After listing breaker details, then list compatible machines as a BULLET LIST.\n"
        "- If there are multiple compatible breakers, list them all. Do not add a 'that's all in dataset' line or similar closing text.\n"
        "- Keep responses concise, human-friendly, and start direct answers with a short lead like: 'Here is the price for Hyundai R30' when answering price queries.\n"
        "- Do not reveal internal normalizations or synonyms. If user typed SB*, simply answer referencing VJ* (without explaining the mapping).\n"
        "- Avoid extra filler lines. Answer to the point.\n\n"
    )

    system_prompt = (
        "You are a strict RAG assistant for YantraLive END-CUSTOMER, SPARE_PARTS, "
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
        "- Include only factual attributes found in the CONTEXT.\n"
        "\n"
        "===================================================\n"
        "COMPARISON QUESTIONS\n"
        "===================================================\n"
        "- Start your answer with: 'Here is a comparison between <X> and <Y>:'\n"
        "- Build a clean Markdown table using this format:\n"
        "  | Feature | Option 1 | Option 2 |\n"
        "  |--------|----------|----------|\n"
        "- Only include facts that are clearly present in the CONTEXT.\n"
        "\n"
        "===================================================\n"
        "SPECIFIC PARAMETER QUESTIONS\n"
        "===================================================\n"
        "- Find the row(s) that match the machine/breaker mentioned.\n"
        "- Extract the exact requested numeric or textual value from those row(s).\n"
        "- If multiple rows give different values, mention each distinct value clearly.\n"
        "- Keep the answer short and directly focused on the requested parameter.\n"
        "\n"
        "===================================================\n"
        "SUBJECTIVE / BEST-OPTION QUESTIONS\n"
        "===================================================\n"
        "- First, list all relevant options as a bullet list with their key specs.\n"
        "- Then, based ONLY on the CONTEXT, choose ONE option as the best.\n"
        "- Clearly say: 'According to the available data, the best option is <NAME> because ...'\n"
        "- Do NOT invent reasons that are not supported by the CONTEXT.\n"
        "\n"
        "REMEMBER\n"
        "- Never invent breakers, dealers, or spare parts that are not present in the CONTEXT.\n"
        "- If the machine, breaker, part, or dealer mentioned is not present at all,\n"
        "  reply UNSURE_FROM_DATA.\n"
    )

    full_system_prompt = prefix_rule + system_prompt

    user_content = f"""
CONTEXT (rows from YantraLive datasets):
{context}

USER QUESTION:
{user_question}
"""

    try:
        messages = [
            {"role": "system", "content": full_system_prompt},
            {"role": "user", "content": user_content},
        ]

        resp = openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
            max_tokens=1500,
        )

        choice = None
        # robust extraction of first choice
        choices = getattr(resp, "choices", resp.get("choices") if isinstance(resp, dict) else None)
        if choices is None:
            choices = resp["choices"]
        if isinstance(choices, (list, tuple)) and len(choices) > 0:
            choice = choices[0]
        else:
            choice = choices

        answer = _extract_choice_message(choice)
        return answer
    except Exception as e:
        print(f"[OPENAI ERROR] {e}")
        return None


# ==========================
# ROUTES
# ==========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msg = req.messages[-1].content
    print(f"[CHAT] Original User message: {user_msg!r}")

    normalized_user_msg = auto_normalize_sb(user_msg)
    if normalized_user_msg != user_msg:
        print(f"[CHAT] Normalized User message (SB->VJ): {normalized_user_msg!r}")
    else:
        print(f"[CHAT] Normalized User message: {normalized_user_msg!r}")

    def is_brand_only(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        if re.search(r"\d", text):
            return False
        if len(text.split()) > 3:
            return False
        return True

    def is_likely_model_value(val: str, brand: str) -> bool:
        if not val:
            return False
        v = val.strip()
        low = v.lower()
        if low == brand.lower():
            return False
        if re.search(r"\bton\b", low) or re.search(r"\btons\b", low) or re.search(r"\btonnage\b", low):
            return False
        if re.search(r"^\d+(\s*-\s*\d+)?\s*(t|ton|tons)?$", low):
            return False
        if re.search(r"\b(vj|sb)\s*\d+", v, re.IGNORECASE) or re.search(r"\bvj\b", v, re.IGNORECASE):
            return False
        if len(v) <= 3 and v.isalpha():
            return False
        if re.fullmatch(r"[A-Za-z\s]+SMART", v, re.IGNORECASE):
            return False
        if len(v) > 80:
            return False
        if brand.lower() in low:
            return True
        if re.search(r"[A-Za-z]+\d", v) or re.search(r"\d+[A-Za-z]+", v) or "-" in v:
            return True
        if re.fullmatch(r"[A-Za-z0-9\-_/]+", v) and (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            return True
        return False

    # 1) Embed the (normalized) query with OpenAI
    try:
        query_vec = embed_query(normalized_user_msg)
    except Exception as e:
        print(f"[ERROR] Failed to embed user query with OpenAI: {e}")
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
                n_results=10,
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

    unique_docs = list(dict.fromkeys(docs))
    context = "\n\n---\n\n".join(unique_docs)

    if is_brand_only(normalized_user_msg):
        brand = normalized_user_msg.strip()
        candidate_models: List[str] = []
        for d in unique_docs:
            pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
            for (k, v) in pairs:
                k_clean = k.strip().lower()
                val = v.strip()
                if "model" in k_clean or "machine" in k_clean or "machine model" in k_clean:
                    if is_likely_model_value(val, brand):
                        candidate_models.append(val)
        if not candidate_models:
            for d in unique_docs:
                parts = re.split(r"[|,/\\\n]+", d)
                for p in parts:
                    token = p.strip()
                    if not token:
                        continue
                    if is_likely_model_value(token, brand):
                        candidate_models.append(token)
        seen = set()
        models = []
        for m in candidate_models:
            m_norm = m.strip()
            m_norm = re.sub(r"^[\W_]+|[\W_]+$", "", m_norm)
            if m_norm and m_norm not in seen:
                if re.search(r"\bton\b", m_norm.lower()):
                    continue
                if re.search(r"^(vj|sb)\b", m_norm, re.IGNORECASE):
                    continue
                if m_norm.lower() == brand.lower():
                    continue
                seen.add(m_norm)
                models.append(m_norm)
        if models:
            md_lines = [f"**Models for {brand}:**"]
            for m in models:
                md_lines.append(f"- {m}")
            md_answer = "\n".join(md_lines)
            return ChatResponse(
                answer=md_answer,
                used_context=unique_docs,
                from_fallback=False,
            )

    # 3) Ask OpenAI to answer using this context
    raw = generate_with_groq(context=context, user_question=normalized_user_msg)
    if raw is None:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    raw = raw.strip()

    # 4) If model says UNSURE_FROM_DATA → human fallback
    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(
            answer=fallback(),
            used_context=unique_docs,
            from_fallback=True,
        )

    # 5) Normal answer — optionally attach brochure_url if relevant
    brochure_url = None

    m = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd|hd$))?\b", normalized_user_msg, re.IGNORECASE)
    if m:
        parts = [m.group(1) or "", m.group(2) or ""]
        if m.group(3):
            parts.append(m.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
        fname = BROCHURE_MAP.get(key_norm)
        if not fname:
            alt_key = re.sub(r"hd$", "", key_norm)
            fname = BROCHURE_MAP.get(alt_key)
        if fname:
            base = str(request.base_url).rstrip("/")
            brochure_url = f"{base}/brochures/view/{fname}"

    if not brochure_url:
        m2 = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", raw, re.IGNORECASE)
        if m2:
            key_raw = "".join([m2.group(1) or "", m2.group(2) or ""] + ([m2.group(3)] if m2.group(3) else []))
            key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
            fname = BROCHURE_MAP.get(key_norm)
            if fname:
                base = str(request.base_url).rstrip("/")
                brochure_url = f"{base}/brochures/view/{fname}"

    return ChatResponse(
        answer=raw,
        used_context=unique_docs,
        from_fallback=False,
        brochure_url=brochure_url,
    )

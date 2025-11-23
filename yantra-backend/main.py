import os
import time
import re
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Serve brochures (static)
# ==========================
BROCHURE_DIR = os.path.join("data", "brochures")
# Ensure folder exists (server will still run if missing)
os.makedirs(BROCHURE_DIR, exist_ok=True)
app.mount("/brochures", StaticFiles(directory=BROCHURE_DIR), name="brochures")

# Build brochure map: normalized key -> filename
def _build_brochure_map():
    m = {}
    if not os.path.isdir(BROCHURE_DIR):
        return m
    for fname in os.listdir(BROCHURE_DIR):
        if not fname.lower().endswith(".pdf"):
            continue
        name_no_ext = os.path.splitext(fname)[0]
        # normalized key: lowercase, remove non-alphanumerics
        key = re.sub(r"[^a-z0-9]", "", name_no_ext.lower())
        m[key] = fname
    return m

BROCHURE_MAP = _build_brochure_map()

# ==========================
# CHROMA (we pass embeddings manually)
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
    brochure_url: Optional[str] = None  # optional brochure link

# ==========================
# COHERE EMBEDDINGS + RETRY
# ==========================
EMBED_MODEL = "embed-english-v3.0"
EMBED_BATCH_SIZE = 64
EMBED_BATCH_SLEEP_SECONDS = 1.0


def _cohere_embed_with_retry(
    texts: List[str],
    input_type: str,
    label: str = "",
    max_retries: int = 5,
) -> List[List[float]]:
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
    return _cohere_embed_with_retry(
        texts=texts,
        input_type="search_document",
        label=f"documents batch (size={len(texts)})",
    )


def embed_query(text: str) -> List[float]:
    embeddings = _cohere_embed_with_retry(
        texts=[text],
        input_type="search_query",
        label="user query",
    )
    return embeddings[0]

# ==========================
# LOAD CSV + INDEX (Cohere -> Chroma)
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
            f"[INDEX] Finished indexing {total_docs} rows for {tag} using Cohere embeddings."
        )
    except Exception as e:
        print(f"[WARN] Failed to embed/index dataset {tag} with Cohere: {e}")
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
# GROQ GENERATION (Llama 3.3) with prefix rule
# ==========================
GROQ_MODEL_ID = "llama-3.3-70b-versatile"


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
        resp = groq_client.chat.completions.create(
            model=GROQ_MODEL_ID,
            messages=[
                {"role": "system", "content": full_system_prompt},
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

    # --- Helper: simple brand-only detector --------------------------------
    def is_brand_only(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        # If the text contains any digit, treat not-brand-only (models often contain digits)
        if re.search(r"\d", text):
            return False
        # If the user typed more than 3 words assume it's not a single brand
        if len(text.split()) > 3:
            return False
        # OK: short, no digits — treat as brand-only
        return True
    # -----------------------------------------------------------------------

    # Helper: decide if a value is likely a machine model (filter out tonnage, breaker SKUs, brand names, etc)
    def is_likely_model_value(val: str, brand: str) -> bool:
        if not val:
            return False
        v = val.strip()
        low = v.lower()

        # Reject pure brand tokens (like "HYUNDAI") — we want model names, not brand label
        if low == brand.lower():
            return False

        # Reject tonnage / ranges (e.g., "16-25 Ton", "7-10 Ton", "30-45 Ton")
        if re.search(r"\bton\b", low) or re.search(r"\btons\b", low) or re.search(r"\btonnage\b", low):
            return False
        if re.search(r"^\d+(\s*-\s*\d+)?\s*(t|ton|tons)?$", low):
            return False

        # Reject obvious breaker SKUs like VJ*, SB* (we only want machine models)
        if re.search(r"\b(vj|sb)\s*\d+", v, re.IGNORECASE) or re.search(r"\bvj\b", v, re.IGNORECASE):
            return False

        # Exclude rows that look like capacity classes or short labels (e.g., '7-10 Ton', 'Loader')
        if len(v) <= 3 and v.isalpha():
            return False

        # Exclude if contains words like 'SMART' alone or too-generic tokens (but allow if combined with model)
        if re.fullmatch(r"[A-Za-z\s]+SMART", v, re.IGNORECASE):
            # likely a variant label, not the model primary name -> skip
            return False

        # Avoid ridiculously long cells that are unlikely a model name
        if len(v) > 80:
            return False

        # If the value contains the brand name or brand abbreviation, it's likely a machine model mention
        if brand.lower() in low:
            return True

        # Otherwise, heuristic: model strings often have letters + digits or hyphen (e.g., R215LC-7, R230)
        if re.search(r"[A-Za-z]+\d", v) or re.search(r"\d+[A-Za-z]+", v) or "-" in v:
            return True

        # Also accept short alphanumeric tokens like "R85", "HX225SL"
        if re.fullmatch(r"[A-Za-z0-9\-_/]+", v) and (re.search(r"[A-Za-z]", v) and re.search(r"\d", v)):
            return True

        return False

    # 1) Embed the (normalized) query with Cohere
    try:
        query_vec = embed_query(normalized_user_msg)
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

    # If the query looks like a brand only -> return only models (no other text)
    if is_brand_only(normalized_user_msg):
        brand = normalized_user_msg.strip()
        candidate_models: List[str] = []

        # Attempt 1: prefer explicit "model" or "machine" fields in the doc (key:value pairs)
        for d in unique_docs:
            # find all key: value pairs separated by ' | ' or line breaks
            pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
            for (k, v) in pairs:
                k_clean = k.strip().lower()
                val = v.strip()
                # If key looks like model/machine/model name, prioritize it
                if "model" in k_clean or "machine" in k_clean or "machine model" in k_clean:
                    if is_likely_model_value(val, brand):
                        candidate_models.append(val)

        # Attempt 2: also scan whole doc text for tokens that look like machine models (fallback)
        if not candidate_models:
            for d in unique_docs:
                # split by separators and pipes and commas
                parts = re.split(r"[|,/\\\n]+", d)
                for p in parts:
                    token = p.strip()
                    # avoid adding the brand label or short generic tokens
                    if not token:
                        continue
                    if is_likely_model_value(token, brand):
                        candidate_models.append(token)

        # dedupe preserving order, and normalize model strings (strip extra whitespace)
        seen = set()
        models = []
        for m in candidate_models:
            m_norm = m.strip()
            # remove trailing/leading punctuation like ':' or '-'
            m_norm = re.sub(r"^[\W_]+|[\W_]+$", "", m_norm)
            if m_norm and m_norm not in seen:
                # final safeguard: skip if it looks like capacity / tonnage or contains 'ton'
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

        # if no models found from docs, fall back to Groq (below) so normal flow applies

    # 3) Ask Groq to answer using this context
    raw = generate_with_groq(context=context, user_question=normalized_user_msg)
    if raw is None:
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

    # 5) Normal answer — optionally attach brochure_url if relevant
    brochure_url = None

    # Heuristic: find a breaker token in normalized_user_msg (VJ... optionally HD)
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
            brochure_url = f"{base}/brochures/{fname}"

    # Another heuristic: sometimes the model is referenced in the raw Groq answer (e.g., "VJ30 HD")
    if not brochure_url:
        m2 = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", raw, re.IGNORECASE)
        if m2:
            key_raw = "".join([m2.group(1) or "", m2.group(2) or ""] + ([m2.group(3)] if m2.group(3) else []))
            key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
            fname = BROCHURE_MAP.get(key_norm)
            if fname:
                base = str(request.base_url).rstrip("/")
                brochure_url = f"{base}/brochures/{fname}"

    return ChatResponse(
        answer=raw,
        used_context=unique_docs,
        from_fallback=False,
        brochure_url=brochure_url,
    )

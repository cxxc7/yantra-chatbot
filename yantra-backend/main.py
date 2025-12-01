import os
import time
import re
from typing import List, Literal, Optional, Any, Set, Tuple

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

# how many sales nudges to present per conversation before backing off
SALES_NUDGE_LIMIT = int(os.getenv("YANTRA_SALES_NUDGE_LIMIT", "2"))

SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI(
    title="YantraLive RAG Chatbot (OpenAI)",
    version="1.3",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================
# Serve brochures (static) + preview wrapper endpoints
# ==========================
BROCHURE_DIR = os.path.join("data", "brochures")
os.makedirs(BROCHURE_DIR, exist_ok=True)
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
# OPENAI EMBEDDINGS + RETRY
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
            data = getattr(resp, "data", resp.get("data") if isinstance(resp, dict) else None)
            if data is None:
                data = resp["data"]
            embeddings = []
            for d in data:
                if isinstance(d, dict):
                    embeddings.append(d.get("embedding"))
                else:
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
# LOAD CSV + INDEX
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
# NORMALIZER
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
# RAG SYSTEM PROMPT & GENERATION (kept as requested)
# ==========================
GROQ_MODEL_ID = "llama-3.3-70b-versatile"  # parity-only tag

def _extract_choice_message(choice: Any) -> str:
    msg = None
    if isinstance(choice, dict):
        msg = choice.get("message") or choice.get("text") or choice.get("content")
    else:
        msg = getattr(choice, "message", None) or getattr(choice, "text", None) or getattr(choice, "content", None)
    if isinstance(msg, dict):
        return msg.get("content") or msg.get("text") or ""
    if isinstance(msg, str):
        return msg
    return getattr(msg, "content", "") if msg is not None else ""


def generate_with_groq(context: str, user_question: str, user_ton: Optional[int] = None) -> Optional[str]:
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
        "TONNAGE HANDLING RULES (use these to match ranges):\n"
        "- When the user provides a numeric tonnage value (e.g., '8 ton', '9 t', '10'), the assistant MUST match that numeric value against any 'capacity' / 'tonnage' fields present in the CONTEXT.\n"
        "- If a CONTEXT row contains a range like '7-10 Ton' then integer values 7, 8, 9, and 10 are considered inside that range.\n"
        "- Accept range formats like '7-10 Ton', '7 to 10 Ton', '7–10 Ton', '7  -  10T', '7T' and treat them inclusively.\n"
        "- If the user requests a single ton value, prefer breakers where the CONTEXT explicitly lists a range or class covering that value. If multiple breakers cover the ton value, list all.\n"
        "- If the CONTEXT doesn't have explicit ranges covering the value, reply EXACTLY: UNSURE_FROM_DATA.\n\n"
        "WARRANTY & STANDARD TERMS (default policy to use when CONTEXT lacks explicit overrides):\n"
        "- Standard warranty: 1-year warranty covering the piston, the piston body, the cylinder, and the front head.\n"
        "- Warranty is conditional on regular maintenance (maintenance mandatory to claim warranty).\n"
        "- Parts INCLUDED under warranty: piston, piston body, front head.\n"
        "- Parts EXCLUDED from warranty: back head, seal, control valve, bushes, chisels, end hose, breaker body, through bolt.\n"
        "- GST: 18% applies on breakers/parts unless a CONTEXT row explicitly shows a different tax rate.\n"
        "- Payment terms (default): 100% advance required before dispatch.\n"
        "- Dispatch ETA (default): within 24 hours of full payment received, unless CONTEXT row overrides.\n"
        "- Standard package inclusions (default): two chisels, one empty gas cylinder, one gas charging kit, two end hoses, one tool kit.\n"
        "- Transport (default): shipped on two-pay basis via VRL by default, or transport of buyer's choice if specified in CONTEXT.\n"
        "- IMPORTANT: Use these warranty/terms only when the CONTEXT does not present explicit, conflicting policy. If CONTEXT contains warranty/terms for the specific model, prefer CONTEXT values. If unclear, reply EXACTLY: UNSURE_FROM_DATA.\n\n"
        "SALES FOLLOW-UP RULES (when user asks about a specific breaker/model):\n"
        "- After answering facts about a specific breaker/model, present a short polite set of follow-up action suggestions as QUESTIONS (e.g., 'Would you like the brochure for VJ20?').\n"
        "- Suggested follow-ups: brochure, connect to dealer/contact for quote, get price / proforma (remember default payment/GST rules), more specs or warranty details.\n"
        "- Respect user's negative response. If the user replies 'no' or indicates disinterest, stop nudging and move on.\n"
        "- Do not be aggressive; keep follow-ups helpful and concise.\n\n"
        "STRICT RAG & RANGE MATCHING RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT rows and the explicit USER_TON when provided.\n"
        "- If the answer is not supported by the CONTEXT and not derivable from USER_TON and CONTEXT rows, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT invent models, breakers, dealers, or parts. Do NOT use outside knowledge except the warranty/terms defaults listed above when CONTEXT lacks them.\n\n"
    )

    system_prompt = (
        "You are a strict RAG assistant for YantraLive END-CUSTOMER, SPARE_PARTS, and DEALER rock breaker data.\n"
        "\n"
        "GENERAL RULES:\n"
        "- You MUST use ONLY the facts from the CONTEXT.\n"
        "- If the answer is not clearly present in the CONTEXT, reply EXACTLY: UNSURE_FROM_DATA.\n"
        "- Do NOT guess. Do NOT use outside knowledge except the explicit warranty & terms defined in the prefixed rules when CONTEXT lacks them.\n"
        "- Keep answers concise, factual, and formatted cleanly.\n"
        "- Respect the dataset tags [END_CUSTOMER], [SPARE_PARTS], [DEALERS] when reasoning.\n"
        "\n"
        "COMPATIBILITY QUESTIONS:\n"
        "- Scan EVERY row in the CONTEXT.\n"
        "- Identify all rows where the machine brand and/or machine model match the user query.\n"
        "- For tonnage queries: match numeric ton values against any capacity/tonnage ranges found in CONTEXT (e.g., '7-10 Ton' includes 7..10).\n"
        "- Extract all DISTINCT compatible breaker models / SKUs and present them in the exact format below.\n"
        "\n"
        "OUTPUT FORMATS:\n"
        "Compatibility:\n"
        "**Compatible Breakers:**\n"
        "- <Breaker Model / SKU> – <key facts from dataset only>\n"
        "- <Breaker Model / SKU> – <key facts from dataset only>\n"
        "\n"
        "Comparison:\n"
        "Start: 'Here is a comparison between <X> and <Y>:' then a markdown table:\n"
        "| Feature | Option 1 | Option 2 |\n"
        "|--------|----------|----------|\n"
        "\n"
        "Specific parameter questions:\n"
        "- Extract the exact requested numeric or textual values. If multiple rows give different values, list each distinct value.\n"
        "\n"
        "SUBJECTIVE / BEST-OPTION QUESTIONS:\n"
        "- List all relevant options as bullets with key specs, then choose ONE option as the best with a short justification supported only by CONTEXT.\n"
        "\n"
        "REMEMBER:\n"
        "- Never invent breakers, dealers, or spare parts that are not present in the CONTEXT.\n"
        "- If the machine, breaker, part, or dealer mentioned is not present at all, reply EXACTLY: UNSURE_FROM_DATA.\n"
    )

    full_system_prompt = prefix_rule + system_prompt

    user_content = f"""
CONTEXT (rows from YantraLive datasets):
{context}

USER QUESTION:
{user_question}
"""
    if user_ton is not None:
        user_content += f"\nUSER_TON: {user_ton}\n"

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

        choices = getattr(resp, "choices", resp.get("choices") if isinstance(resp, dict) else None)
        if choices is None:
            choices = resp["choices"]
        choice = choices[0] if isinstance(choices, (list, tuple)) and len(choices) > 0 else choices
        answer = _extract_choice_message(choice)
        return answer
    except Exception as e:
        print(f"[OPENAI ERROR] {e}")
        return None


# ==========================
# Helper deterministic extraction utilities
# ==========================
def _find_vj_tokens_in_text(text: str) -> Set[str]:
    if not text:
        return set()
    tokens = set(re.findall(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", text, flags=re.IGNORECASE))
    out = set()
    for t in tokens:
        joined = "".join([p for p in t if p])
        normalized = re.sub(r"[^a-z0-9]", "", joined.lower())
        out.add(normalized)
    return out


def _extract_field_values_from_row(row_text: str, field_names: List[str]) -> dict:
    pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", row_text)
    d = {}
    for (k, v) in pairs:
        d[k.strip().lower()] = v.strip()
    res = {}
    for fn in field_names:
        res[fn] = d.get(fn.lower())
    return res


def _collect_all_compatible_machines_or_breakers(query_token: str, docs: List[str]) -> Tuple[Set[str], List[str]]:
    found: Set[str] = set()
    source_rows: List[str] = []
    qlow = query_token.lower()
    for d in docs:
        low = d.lower()
        if qlow in low:
            source_rows.append(d)
            pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
            for (k, v) in pairs:
                k_clean = k.strip().lower()
                if "compatible" in k_clean:
                    tokens = re.split(r"[,;/\\\|]+", v)
                    for t in tokens:
                        t = t.strip()
                        if not t:
                            continue
                        vj_matches = re.findall(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", t, flags=re.IGNORECASE)
                        if vj_matches:
                            for m in vj_matches:
                                joined = "".join([p for p in m if p])
                                normalized = re.sub(r"[^a-z0-9]", "", joined.lower())
                                found.add(normalized)
                        else:
                            norm = re.sub(r"[^a-z0-9 ]", "", t.lower())
                            if norm:
                                found.add(norm)
            vjset = _find_vj_tokens_in_text(d)
            found.update(vjset)
    return found, source_rows


def _is_simple_negative_reply(text: str) -> bool:
    if not text:
        return False
    text = text.strip().lower()
    negatives = {"no", "nope", "nah", "not now", "dont want", "don't want", "no thanks", "no thank you", "n"}
    if text in negatives:
        return True
    if re.fullmatch(r"no[.!?]?", text):
        return True
    return False


def _count_prior_followups(messages: List[dict]) -> int:
    count = 0
    for m in messages:
        if m.get("role") == "assistant" and "Would you like any of the following for this model?" in (m.get("content") or ""):
            count += 1
    return count


# ==========================
# ROUTES
# ==========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


def _extract_ton_from_text(text: str) -> Optional[int]:
    if not text:
        return None
    # robust patterns: single "8 ton", "8t", "8 T", numbers with 'ton(s)', or plain numeric followed by context 'ton'
    m = re.search(r"\b(\d{1,3})\s*(?:t(?:on)?s?|tonnes?|tons?)\b", text, re.IGNORECASE)
    if m:
        try:
            return int(m.group(1))
        except:
            return None
    # plain "8T" or "8T" style with optional spaces
    m2 = re.search(r"\b(\d{1,3})\s*[tT]\b", text)
    if m2:
        try:
            return int(m2.group(1))
        except:
            return None
    # sometimes user may just type a number with 'ton' implied but we avoid over-triggering
    return None


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request):
    if not req.messages:
        raise HTTPException(status_code=400, detail="No messages provided")

    user_msg_raw = req.messages[-1].content
    print(f"[CHAT] Original User message: {user_msg_raw!r}")

    normalized_user_msg = auto_normalize_sb(user_msg_raw)
    if normalized_user_msg != user_msg_raw:
        print(f"[CHAT] Normalized User message (SB->VJ): {normalized_user_msg!r}")
    else:
        print(f"[CHAT] Normalized User message: {normalized_user_msg!r}")

    last_user = normalized_user_msg.strip()

    msgs_for_state = [m.dict() for m in req.messages]
    prior_followups = _count_prior_followups(msgs_for_state)
    print(f"[CHAT] Prior followups presented so far: {prior_followups}")

    if len(req.messages) >= 2 and req.messages[-2].role == "assistant":
        prev_assistant_text = req.messages[-2].content or ""
        if "Would you like any of the following for this model?" in prev_assistant_text:
            low = last_user.lower().strip()
            if _is_simple_negative_reply(low):
                reply_text = "No problem — I won't suggest that again right now. Anything else I can help you with?"
                return ChatResponse(answer=reply_text, used_context=[], from_fallback=False)

            if any(k in low for k in ("brochure", "open brochure", "pdf", "open the brochure")):
                m = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", prev_assistant_text + " " + normalized_user_msg, re.IGNORECASE)
                if m:
                    parts = [m.group(1) or "", m.group(2) or ""]
                    if m.group(3):
                        parts.append(m.group(3))
                    key_raw = "".join(parts)
                    key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
                    fname = BROCHURE_MAP.get(key_norm) or BROCHURE_MAP.get(re.sub(r"hd$", "", key_norm))
                    if fname:
                        base = str(request.base_url).rstrip("/")
                        brochure_url = f"{base}/brochures/view/{fname}"
                        reply = f"I've opened the brochure for {m.group(0)}. [📘 Open Brochure]({brochure_url})\n\nWould you like me to connect you with a dealer or get you a quote?"
                        return ChatResponse(answer=reply, used_context=[], from_fallback=False, brochure_url=brochure_url)
                return ChatResponse(answer="I couldn't find a brochure for that model in our files.", used_context=[], from_fallback=False)

            if any(k in low for k in ("connect", "contact", "dealer", "quote", "connect me")):
                reply = (
                    f"I can connect you with a dealer. Please share your preferred contact number or email and preferred city.\n\n"
                    f"Or I can share our support contact: 📞 {SUPPORT_PHONE} • 📧 {SUPPORT_EMAIL}"
                )
                return ChatResponse(answer=reply, used_context=[], from_fallback=False)

            if any(k in low for k in ("price", "quote", "proforma", "cost", "how much")):
                reply = (
                    "I can prepare a price/proforma. Default terms: 100% advance, GST 18% applies, dispatch ETA within 24 hours of payment. "
                    "Please confirm the model you'd like a quote for (e.g., VJ20 HD) so I can fetch the exact price."
                )
                return ChatResponse(answer=reply, used_context=[], from_fallback=False)

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

    try:
        query_vec = embed_query(normalized_user_msg)
    except Exception as e:
        print(f"[ERROR] Failed to embed user query with OpenAI: {e}")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

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

    vj_match_user = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", normalized_user_msg, re.IGNORECASE)
    if vj_match_user:
        parts = [vj_match_user.group(1) or "", vj_match_user.group(2) or ""]
        if vj_match_user.group(3):
            parts.append(vj_match_user.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
        compat_set, src_rows = _collect_all_compatible_machines_or_breakers(key_norm, unique_docs)
        if compat_set:
            lines = [f"**Compatible machines / breakers for {key_raw.upper()}:**"]
            for c in sorted(compat_set):
                if re.fullmatch(r"vj\d+hd?", c):
                    display = c.upper()
                else:
                    display = c
                lines.append(f"- {display}")
            followup_text = ""
            if prior_followups < SALES_NUDGE_LIMIT:
                followup_text = (
                    "\n\n**Would you like any of the following for this model?**\n"
                    "- Brochure (reply 'brochure')\n"
                    "- Connect me with a dealer/contact for a quote (reply 'connect')\n"
                    "- Get a price / proforma (reply 'price')\n"
                    "- More specs or warranty details (reply 'specs' or 'warranty')\n\n"
                    "Reply with which option you'd like, or 'no' to skip."
                )
            reply_text = "\n".join(lines) + followup_text
            brochure_url = None
            fname = BROCHURE_MAP.get(key_norm)
            if fname:
                base = str(request.base_url).rstrip("/")
                brochure_url = f"{base}/brochures/view/{fname}"
            return ChatResponse(answer=reply_text, used_context=src_rows or unique_docs, from_fallback=False, brochure_url=brochure_url)

    user_ton = _extract_ton_from_text(normalized_user_msg)
    if user_ton is not None:
        matched_breakers: Set[str] = set()
        matched_rows: List[str] = []
        for d in unique_docs:
            pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
            capacity_fields = []
            for (k, v) in pairs:
                if re.search(r"capacit|tonnage|ton\b|tons\b|class", k, re.IGNORECASE):
                    capacity_fields.append((k.strip(), v.strip()))
            if capacity_fields:
                for (k, v) in capacity_fields:
                    v_clean = v.replace("\u2013", "-").replace("\u2014", "-")  # normalize dashes
                    # normalize "to" forms into hyphen for consistent parsing
                    v_clean = re.sub(r"\bto\b", "-", v_clean, flags=re.IGNORECASE)
                    # find ranges like 7-10 or 7 - 10, optionally with T/Ton
                    rmatches = re.findall(r"(\d{1,3})\s*(?:-|–|—)\s*(\d{1,3})", v_clean)
                    matched = False
                    for rmatch in rmatches:
                        a = int(rmatch[0]); b = int(rmatch[1])
                        if a <= user_ton <= b:
                            matched_rows.append(d)
                            matched_breakers.update(_find_vj_tokens_in_text(d))
                            matched = True
                            break
                    if matched:
                        continue
                    # single number like '8 Ton' or '8T' inside capacity field
                    sm = re.search(r"\b(\d{1,3})\b", v)
                    if sm and int(sm.group(1)) == user_ton:
                        matched_rows.append(d)
                        matched_breakers.update(_find_vj_tokens_in_text(d))
            else:
                # fallback: examine raw row text for range patterns
                text_norm = d.replace("\u2013", "-").replace("\u2014", "-")
                text_norm = re.sub(r"\bto\b", "-", text_norm, flags=re.IGNORECASE)
                rmatches = re.findall(r"(\d{1,3})\s*(?:-|–|—)\s*(\d{1,3})", text_norm)
                for rmatch in rmatches:
                    a = int(rmatch[0]); b = int(rmatch[1])
                    if a <= user_ton <= b:
                        matched_rows.append(d)
                        matched_breakers.update(_find_vj_tokens_in_text(d))
                        break
        if matched_breakers:
            lines = [f"**Breakers compatible with {user_ton} Ton machines:**"]
            for b in sorted(matched_breakers):
                lines.append(f"- {b.upper()}")
            followup_text = ""
            if prior_followups < SALES_NUDGE_LIMIT:
                followup_text = (
                    "\n\n**Would you like any of the following?**\n"
                    "- Brochure (reply 'brochure')\n"
                    "- Connect me with a dealer/contact for a quote (reply 'connect')\n"
                    "- Get a price / proforma (reply 'price')\n"
                    "- More specs or warranty details (reply 'specs' or 'warranty')\n\n"
                    "Reply with which option you'd like, or 'no' to skip."
                )
            return ChatResponse(answer="\n".join(lines) + followup_text, used_context=matched_rows or unique_docs, from_fallback=False)

    raw = generate_with_groq(context=context, user_question=normalized_user_msg, user_ton=_extract_ton_from_text(normalized_user_msg))
    if raw is None:
        return ChatResponse(answer=fallback(), used_context=unique_docs, from_fallback=True)

    raw = raw.strip()

    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(answer=fallback(), used_context=unique_docs, from_fallback=True)

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

    final_answer = raw
    if prior_followups < SALES_NUDGE_LIMIT:
        if brochure_url or re.search(r"\b(vj)\s*\d{1,3}", raw, re.IGNORECASE):
            followups_text = (
                "\n\n**Would you like any of the following for this model?**\n"
                "- Brochure (reply 'brochure')\n"
                "- Connect me with a dealer/contact for a quote (reply 'connect')\n"
                "- Get a price / proforma (reply 'price')\n"
                "- More specs or warranty details (reply 'specs' or 'warranty')\n\n"
                "Reply with which option you'd like, or 'no' to skip."
            )
            final_answer = f"{final_answer}{followups_text}"

    if brochure_url:
        final_answer = f"{final_answer}\n\n[📘 Open Brochure]({brochure_url})"

    return ChatResponse(answer=final_answer, used_context=unique_docs, from_fallback=False, brochure_url=brochure_url)

# main.py
import os
import time
import re
import shutil
import tempfile
from typing import List, Literal, Optional, Any, Set, Tuple, Dict

from fastapi import FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

import pandas as pd
import chromadb
from openai import OpenAI

from datetime import datetime
from zoneinfo import ZoneInfo

# ==========================
# ENVIRONMENT
# ==========================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

# translation model can be overridden if you want a cheaper model
TRANSLATE_MODEL = os.getenv("YANTRA_TRANSLATE_MODEL", OPENAI_MODEL)

# how many sales nudges to present per conversation before backing off
SALES_NUDGE_LIMIT = int(os.getenv("YANTRA_SALES_NUDGE_LIMIT", "2"))

SUPPORT_PHONE = os.getenv("YANTRALIVE_SUPPORT_PHONE", "+91-9876543210")
SUPPORT_EMAIL = os.getenv("YANTRALIVE_SUPPORT_EMAIL", "support@yantralive.com")

# STT config
YANTRA_STT_MAX_BYTES = int(os.getenv("YANTRA_STT_MAX_BYTES", str(12 * 1024 * 1024)))  # 12MB default
YANTRA_STT_TMP_DIR = os.getenv("YANTRA_STT_TMP_DIR", None)  # optional custom tmp dir

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

# ==========================
# FASTAPI
# ==========================
app = FastAPI(
    title="YantraLive RAG Chatbot (OpenAI)",
    version="1.5",
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
    followup_meta: Optional[Dict[str, Any]] = None


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

    # use robust CSV loading
    try:
        df = pd.read_csv(path, encoding="utf-8", dtype=str).fillna("")
    except Exception:
        # fallback to default pandas guessing if encoding utf-8 fails
        df = pd.read_csv(path, dtype=str).fillna("")

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
    # IMPORTANT: Keep startup resilient — indexing runs synchronously here but errors are handled.
    load_all_datasets()
except Exception as e:
    print(f"[WARN] Dataset indexing failed on startup: {e}")


# ==========================
# FALLBACK MESSAGE / DEFAULTS
# ==========================
DEFAULT_WARRANTY_TEXT = (
    "Standard warranty: 1-year warranty covering the piston, the piston body, the cylinder, and the front head.\n"
    "Warranty is conditional on regular maintenance (maintenance mandatory to claim warranty).\n"
    "Parts INCLUDED under warranty: piston, piston body, front head.\n"
    "Parts EXCLUDED from warranty: back head, seal, control valve, bushes, chisels, end hose, breaker body, through bolt.\n"
)
DEFAULT_GST_TEXT = "GST: 18% applies on breakers/parts unless a CONTEXT row explicitly shows a different tax rate."

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
# Translation helper (to English)
# ==========================
def _translate_to_english(text: str) -> str:
    """
    Translate the input text to English using the OpenAI chat completion endpoint.
    Returns empty string on failure.
    """
    if not text:
        return ""
    try:
        # Keep translation prompt short and deterministic
        sys_prompt = "You are a translation helper. Translate the user's message to fluent, plain English only. Do not add any commentary."
        user_prompt = f"Translate to English (only the translation, no extra text):\n\n{text}"
        resp = openai_client.chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=256,
        )
        choices = getattr(resp, "choices", resp.get("choices") if isinstance(resp, dict) else None)
        if not choices:
            return ""
        choice = choices[0] if isinstance(choices, (list, tuple)) else choices
        # extract content safely
        msg = None
        if isinstance(choice, dict):
            msg = choice.get("message") or choice.get("text") or choice.get("content")
        else:
            msg = getattr(choice, "message", None) or getattr(choice, "text", None) or getattr(choice, "content", None)
        if isinstance(msg, dict):
            return (msg.get("content") or msg.get("text") or "").strip()
        if isinstance(msg, str):
            return msg.strip()
        return ""
    except Exception as e:
        print(f"[TRANSLATE] translation failed: {e}")
        return ""


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
        f"- {DEFAULT_WARRANTY_TEXT.replace(chr(10), chr(10) + '- ')}\n"
        f"- {DEFAULT_GST_TEXT}\n"
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
    """
    Keep existing collector for compatibility detection — returns set of tokens and source rows.
    """
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
    negatives = {"no", "nope", "nah", "not now", "dont want", "don't want", "no thanks", "no thank you", "n", "नहीं", "ना", "न"}
    if text in negatives:
        return True
    if re.fullmatch(r"no[.!?]?", text):
        return True
    # also accept some common hindi/kannada negatives (basic)
    if text in {"नहीं", "ना", "ಇಲ್ಲ"}:
        return True
    return False


def _count_prior_followups(messages: List[dict]) -> int:
    count = 0
    for m in messages:
        if m.get("role") == "assistant" and "Would you like any of the following" in (m.get("content") or ""):
            count += 1
    return count


def _parse_followup_comment(text: str) -> Optional[Dict[str, Any]]:
    """
    Parse an HTML comment tag we add to answers, e.g.:
    <!--MODEL_CTX:vj20hd|OPTIONS:brochure,connect,price,specs-->
    Returns dict {model: 'vj20hd', options: ['brochure','connect',...']} or None.
    """
    if not text:
        return None
    m = re.search(r"<!--\s*MODEL_CTX:([a-z0-9]+)\|OPTIONS:([a-z0-9,]+)\s*-->", text, flags=re.IGNORECASE)
    if not m:
        return None
    model = m.group(1).lower()
    opts = [o.strip() for o in m.group(2).split(",") if o.strip()]
    return {"model": model, "options": opts}


def _find_price_for_model(key_norm: str, docs: List[str]) -> Optional[str]:
    # look for common price-related fields in the docs that match the model
    price_keys = ["price", "mrp", "price (in", "price incl", "price including", "price excluding", "minimum price", "price incl gst", "price without gst", "price including gst"]
    found_prices: Set[str] = set()
    for d in docs:
        if key_norm not in d.lower():
            continue
        pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
        for (k, v) in pairs:
            k_clean = k.strip().lower()
            for pk in price_keys:
                if pk in k_clean:
                    found_prices.add(v.strip())
    if found_prices:
        return "; ".join(sorted(found_prices))
    return None


def _query_collection(coll, label: str, query_vec, n_results=50):
    """
    Unified query wrapper. Increased n_results for broader results (so "all compatible" works better).
    """
    try:
        result = coll.query(
            query_embeddings=[query_vec],
            n_results=n_results,
        )
        # return list of documents (flatten)
        docs = []
        try:
            # new style returns dict with 'documents'
            docs_list = result.get("documents", [])
            # documents may be list of lists per query
            if docs_list and isinstance(docs_list[0], list):
                docs = docs_list[0]
            else:
                docs = docs_list or []
        except Exception:
            # fallback defensive handling
            if isinstance(result, dict) and "documents" in result:
                docs = result["documents"][0] if result["documents"] else []
    except Exception as e:
        print(f"[ERROR] Failed to query {label} collection: {e}")
        return []
    return docs


# ==========================
# New helpers to produce the requested output formatting for breaker queries
# ==========================
def _extract_pairs_dict_from_row(row_text: str) -> Dict[str, str]:
    """
    Extract key:value pairs from a row string and return a dict with normalized (lower) keys.
    """
    pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", row_text)
    d: Dict[str, str] = {}
    for (k, v) in pairs:
        k_clean = k.strip()
        v_clean = v.strip()
        if k_clean:
            d[k_clean.lower()] = v_clean
    return d


def _merge_rows_to_field_map(rows: List[str]) -> Dict[str, str]:
    """
    Given multiple dataset row-strings that mention the same breaker, merge fields into one dict.
    Preference: keep the first non-empty value seen for a key.
    """
    merged: Dict[str, str] = {}
    for r in rows:
        d = _extract_pairs_dict_from_row(r)
        for k, v in d.items():
            if not v:
                continue
            if k not in merged or not merged[k]:
                merged[k] = v
    return merged


def _extract_machine_brand_model_from_row_dict(row_dict: Dict[str, str]) -> Optional[str]:
    """
    Given a row dict (lowercase keys), extract a 'Brand - Model' string if possible.
    Looks for common keys like 'machine brand' and 'machine model' or variants.
    """
    brand_keys = ["machine brand", "brand", "maker"]
    model_keys = ["machine model", "model", "machine", "machine name"]
    brand = None
    model = None
    for bk in brand_keys:
        if bk in row_dict and row_dict[bk].strip():
            brand = row_dict[bk].strip()
            break
    for mk in model_keys:
        if mk in row_dict and row_dict[mk].strip():
            model = row_dict[mk].strip()
            break
    if brand and model:
        # sanitize whitespace
        return f"{brand} - {model}"
    # if brand present only, return brand
    if brand and not model:
        return brand
    # if model present only, return model
    if model and not brand:
        return model
    return None


def _parse_compatible_machine_tokens_from_rows(rows: List[str], all_docs: Optional[List[str]] = None, model_keys: Optional[Set[str]] = None) -> List[str]:
    """
    Extract compatible machines as 'Brand - Model' from:
      * explicit 'compatible' fields in the provided rows
      * machine brand/model fields in the provided rows
      * any other docs in all_docs that reference the model(s) in model_keys
    model_keys: set of normalized tokens (e.g., {'vj20','vj20hd'}) to look for when scanning all_docs.
    """
    machines_set: Set[str] = set()

    # Helper to clean a token and skip VJ tokens (breakers)
    def _clean_token(tok: str) -> Optional[str]:
        t = tok.strip()
        if not t:
            return None
        # skip VJ tokens (those are breakers)
        if re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", t, flags=re.IGNORECASE):
            return None
        t = re.sub(r"\s{2,}", " ", t)
        return t.strip()

    # 1) Extract explicit compatible-like keys and machine brand/model in the rows passed in
    for r in rows:
        d = _extract_pairs_dict_from_row(r)
        # machine brand/model direct extraction
        mm = _extract_machine_brand_model_from_row_dict(d)
        if mm:
            machines_set.add(mm)

        # explicit compatible fields
        for k, v in d.items():
            if "compatible" in k or "compatible with" in k or "compatible machines" in k or "compatible models" in k:
                tokens = re.split(r"[,;/\\\|]+", v)
                for t in tokens:
                    cleaned = _clean_token(t)
                    if not cleaned:
                        continue
                    # if cleaned has spaces, split brand & model heuristically
                    if " " in cleaned:
                        parts = cleaned.split()
                        brand = parts[0].strip()
                        model = " ".join(parts[1:]).strip()
                        if brand and model:
                            machines_set.add(f"{brand} - {model}")
                            continue
                    machines_set.add(cleaned)

    # 2) If all_docs provided, scan it for rows that mention the model_keys and collect their machine brand/model fields
    if all_docs and model_keys:
        mk_lower = {k.lower() for k in model_keys}
        for drow in all_docs:
            low = drow.lower()
            # check if any model_key appears in the row
            if any(k in low for k in mk_lower):
                row_dict = _extract_pairs_dict_from_row(drow)
                mm = _extract_machine_brand_model_from_row_dict(row_dict)
                if mm:
                    machines_set.add(mm)
                # also look for explicit compatible fields inside these docs
                for k, v in row_dict.items():
                    if "compatible" in k:
                        tokens = re.split(r"[,;/\\\|]+", v)
                        for t in tokens:
                            cleaned = _clean_token(t)
                            if not cleaned:
                                continue
                            if " " in cleaned:
                                parts = cleaned.split()
                                brand = parts[0].strip()
                                model = " ".join(parts[1:]).strip()
                                if brand and model:
                                    machines_set.add(f"{brand} - {model}")
                                    continue
                            machines_set.add(cleaned)

    # 3) Normalize and return sorted list for deterministic order
    final = sorted({re.sub(r"\s+", " ", m).strip() for m in machines_set})
    return final


# ==========================
# ROUTES
# ==========================
@app.get("/api/health")
def health():
    return {"status": "ok"}


# Helper for greetings & time-based salutation
def _is_greeting(text: str) -> bool:
    if not text:
        return False
    t = text.strip().lower()
    greetings = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "gm", "hello!", "hi!", "hey!", "thanks", "thank you", "thankyou",
        # some localized greetings (basic)
        "नमस्ते", "नमस्कार", "हैलो", "हाय", "ಹೇಗಿದ್ದೀರಾ", "ஹலோ"
    }
    # allow startswith checks for e.g. "good morning everyone"
    for g in greetings:
        if t == g or t.startswith(g + " ") or t.startswith(g + "!") or t.startswith(g + "."):
            return True
    return False


def _time_based_greeting() -> str:
    # Asia/Kolkata timezone per project settings
    try:
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
    except Exception:
        now = datetime.utcnow()
    hour = now.hour
    if 5 <= hour < 12:
        return "Good morning"
    if 12 <= hour < 17:
        return "Good afternoon"
    if 17 <= hour < 22:
        return "Good evening"
    return "Hello"


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


@app.post("/api/stt")
async def stt_endpoint(file: UploadFile = File(...), lang: Optional[str] = "en"):
    """
    POST /api/stt?lang=<en|hi|kn>
    - file: form-data upload (UploadFile)
    - returns JSON: { "transcript": "..." }
    """
    # Basic checks
    if not file:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file uploaded")

    # sanitize filename
    filename = os.path.basename(file.filename or "upload")
    # stream size guard: read header chunk to ensure content-type is audio and not large
    # Note: UploadFile.file is a SpooledTemporaryFile — we'll stream to a temp file and check size
    tmp_dir = YANTRA_STT_TMP_DIR or None
    tmp_file = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1] or ".webm", dir=tmp_dir) as tf:
            tmp_file = tf.name
            total = 0
            # stream write
            while True:
                chunk = await file.read(1024 * 64)
                if not chunk:
                    break
                total += len(chunk)
                if total > YANTRA_STT_MAX_BYTES:
                    # cleanup and abort
                    tf.close()
                    try:
                        os.unlink(tmp_file)
                    except Exception:
                        pass
                    raise HTTPException(status_code=413, detail=f"Uploaded file exceeds max allowed size ({YANTRA_STT_MAX_BYTES} bytes)")
                tf.write(chunk)
            tf.flush()

        # call OpenAI transcription (whisper-1)
        # We let the model auto-detect format; we forward the 'language' parameter when provided
        transcript_text = None
        try:
            with open(tmp_file, "rb") as fh:
                # The OpenAI SDK wrapper used here: openai_client.audio.transcriptions.create
                # using model "whisper-1" which is commonly available
                kwargs = {"file": fh, "model": "whisper-1"}
                if lang:
                    kwargs["language"] = lang
                resp = openai_client.audio.transcriptions.create(**kwargs)
                # resp may be object-like or dict-like; read 'text' or 'transcript'
                transcript_text = None
                if isinstance(resp, dict):
                    transcript_text = resp.get("text") or resp.get("transcript") or ""
                else:
                    # attempt getattr
                    transcript_text = getattr(resp, "text", None) or getattr(resp, "transcript", None) or ""
        except Exception as e:
            print(f"[STT] OpenAI transcription error: {e}")
            raise HTTPException(status_code=500, detail="Transcription failed: " + str(e))

        if transcript_text is None:
            transcript_text = ""

        return JSONResponse({"transcript": transcript_text})
    finally:
        try:
            if tmp_file and os.path.exists(tmp_file):
                os.unlink(tmp_file)
        except Exception:
            pass


# ==========================
# /api/chat implementation (updated for multilingual keyword checks)
# ==========================
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

    # Create a translated-to-English variant for keyword checks
    low = last_user.lower().strip()
    low_en = ""
    try:
        low_en = _translate_to_english(last_user) or ""
        low_en = low_en.lower().strip()
    except Exception as e:
        print(f"[CHAT] Translation attempt failed: {e}")
        low_en = ""
    # Combine both original and translated text for keyword detection
    low_both = " ".join(filter(None, [low, low_en]))

    # Immediately handle greetings (do not proceed to RAG)
    # Use either the original language or the translated English to detect greetings
    if _is_greeting(low) or (low_en and _is_greeting(low_en)):
        sal = _time_based_greeting()
        # preserve short polite form for thanks
        if any(tok in low_both for tok in ("thank", "thanks", "thankyou", "thanks!", "thank you", "धन्यवाद", "धन्यवाद्")):
            return ChatResponse(answer="You're welcome! Anything else I can help with?", used_context=[], from_fallback=False)
        return ChatResponse(answer=f"{sal}! How can I help you today?", used_context=[], from_fallback=False)

    msgs_for_state = [m.dict() for m in req.messages]
    prior_followups = _count_prior_followups(msgs_for_state)
    print(f"[CHAT] Prior followups presented so far: {prior_followups}")

    # If previous assistant offered followups and included the hidden MODEL_CTX comment, extract it
    if len(req.messages) >= 2 and req.messages[-2].role == "assistant":
        prev_assistant_text = req.messages[-2].content or ""
        followup_meta = _parse_followup_comment(prev_assistant_text)
        # Simple negative handling unchanged but multilingual-aware
        if "Would you like any of the following for this model?" in prev_assistant_text:
            if _is_simple_negative_reply(low_both):
                reply_text = "No problem — I won't suggest that again right now. Anything else I can help you with?"
                return ChatResponse(answer=reply_text, used_context=[], from_fallback=False)

        if followup_meta:
            # followup flow: prefer explicit option keywords, then handle simple affirmatives by asking for which option
            model_key = followup_meta.get("model")
            options = followup_meta.get("options", [])

            # Standard option keywords mapping (include some localized tokens)
            option_keywords = {
                "brochure": ["brochure", "pdf", "open brochure", "brochar", "ब्रोशर", "ब्रॉशर"],
                "connect": ["connect", "dealer", "contact", "quote", "connect me", "dealer se jod", "कनेक्ट", "कनेक्ट करें"],
                "price": ["price", "quote", "proforma", "cost", "how much", "dhaam", "daam", "कितना", "दर", "कीमत", "मूल्य"],
                "specs": ["specs", "warranty", "specification", "details", "warranty details", "वारंटी", "वॉरंटी", "विवरण", "स्पेस"],
            }

            # If user provided a clear option request (use low_both)
            for opt, keywords in option_keywords.items():
                if any(k in low_both for k in keywords) and opt in options:
                    # handle same as previous code paths
                    if opt == "brochure":
                        fname = BROCHURE_MAP.get(model_key) or BROCHURE_MAP.get(re.sub(r"hd$", "", model_key))
                        if fname:
                            base = str(request.base_url).rstrip("/")
                            brochure_url = f"{base}/brochures/view/{fname}"
                            reply = f"I've opened the brochure for {model_key.upper()}. [📘 Open Brochure]({brochure_url})\n\nWould you like me to connect you with a dealer or get you a quote?"
                            return ChatResponse(answer=reply, used_context=[], from_fallback=False, brochure_url=brochure_url)
                        return ChatResponse(answer="I couldn't find a brochure for that model in our files.", used_context=[], from_fallback=False)

                    if opt == "connect":
                        reply = (
                            f"I can connect you with a dealer. Please share your preferred contact number or email and preferred city.\n\n"
                            f"Or I can share our support contact: 📞 {SUPPORT_PHONE} • 📧 {SUPPORT_EMAIL}"
                        )
                        return ChatResponse(answer=reply, used_context=[], from_fallback=False)

                    if opt == "price":
                        try:
                            query_vec = embed_query(model_key)
                        except Exception as e:
                            print(f"[ERROR] Failed to embed model query for price: {e}")
                            return ChatResponse(answer=fallback(), used_context=[], from_fallback=True)

                        docs: List[str] = []
                        docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER", query_vec))
                        docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS", query_vec))
                        docs.extend(_query_collection(dealer_collection, "DEALERS", query_vec))

                        price = _find_price_for_model(model_key, docs)
                        if price:
                            reply = f"Here is the price information I found for {model_key.upper()}: {price}\n\nWould you like a proforma/quote for this model?"
                            return ChatResponse(answer=reply, used_context=docs or [], from_fallback=False)
                        else:
                            reply = (
                                f"I couldn't find a direct price record for {model_key.upper()} in our dataset. "
                                "I can prepare a quote/proforma for you — please share your preferred contact details or I can connect you with a dealer."
                            )
                            return ChatResponse(answer=reply, used_context=docs or [], from_fallback=False)

                    if opt == "specs":
                        # call RAG generator for warranty/specs specifically
                        try:
                            query_vec = embed_query(model_key)
                        except Exception as e:
                            print(f"[ERROR] Failed to embed model query for specs: {e}")
                            return ChatResponse(answer=fallback(), used_context=[], from_fallback=True)

                        docs: List[str] = []
                        docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER", query_vec))
                        docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS", query_vec))
                        docs.extend(_query_collection(dealer_collection, "DEALERS", query_vec))

                        raw = generate_with_groq(context="\n\n---\n\n".join(docs), user_question=f"Provide warranty/specs for {model_key.upper()}.")
                        if raw is None or "UNSURE_FROM_DATA" in raw:
                            return ChatResponse(answer=DEFAULT_WARRANTY_TEXT, used_context=docs or [], from_fallback=False)
                        return ChatResponse(answer=raw.strip(), used_context=docs or [], from_fallback=False)

            # If user replied with a simple affirmative (yes/ok/sure) — ask which option (do not run brand-only shortcut)
            affirmatives = {"yes", "y", "yeah", "yep", "sure", "ok", "okay", "please do", "please", "हाँ", "हां", "ठीक है", "ठीक"}
            if low_both in affirmatives or any(low_both.strip().startswith(a) for a in affirmatives):
                return ChatResponse(answer="Sure — which option would you like? You can reply with: 'brochure', 'connect', 'price', or 'specs/warranty'.", used_context=[], from_fallback=False)

            # If user replied with unrelated short replies (single word stopwords), avoid treating them as brand queries
            short_stopwords = {"hi", "hello", "hey", "yes", "no", "ok", "okay", "thanks", "thank", "thankyou", "धन्यवाद", "धन्य", "thank you"}
            if low_both in short_stopwords:
                return ChatResponse(answer="I didn't catch that — which option would you like for this model? (brochure / connect / price / specs)", used_context=[], from_fallback=False)

            # If we reach here, no explicit followup action matched — fallthrough to normal processing (but still keep followup_meta available)
            # (This ensures user can ask a specific question after a followup prompt.)
            # Do not return here; allow normal RAG flow below.

    # helper: detect if the user is asking for warranty/gst style queries and provide defaults when absent
    warranty_keywords = ["warranty", "warranties", "warranty policy", "warranty terms", "standard warranty", "warranty period", "वारंटी", "वारंटी अवधि", "वॉरंटी", "ವಾರೆಂಟಿ"]
    gst_keywords = ["gst", "tax", "vat", "gst percentage", "gst percent", "gst %", "जीएसटी", "कर", "%"]

    # If the user asks about warranty or GST, try to extract from context first; if not present, return defaults
    if any(k in low_both for k in warranty_keywords + gst_keywords):
        # embed the user message and query for context
        try:
            query_vec = embed_query(normalized_user_msg)
        except Exception as e:
            print(f"[ERROR] Failed to embed user query for warranty/gst: {e}")
            return ChatResponse(answer=fallback(), used_context=[], from_fallback=True)

        docs: List[str] = []
        docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER", query_vec))
        docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS", query_vec))
        docs.extend(_query_collection(dealer_collection, "DEALERS", query_vec))

        # Search docs for explicit warranty or gst info
        doc_combined = "\n\n".join(docs).lower()
        found_warranty = "warrant" in doc_combined or "warranty" in doc_combined
        # Defensive check for GST: treat presence of 'gst' or a percent sign as an indicator in text
        if isinstance(doc_combined, str):
            found_gst = ("gst" in doc_combined) or ("%" in doc_combined)
        else:
            found_gst = "gst" in doc_combined

        answer_lines = []
        if any(k in low_both for k in warranty_keywords):
            if found_warranty:
                # call the main model to produce a precise answer from context (honor RAG)
                raw = generate_with_groq(context="\n\n---\n\n".join(docs), user_question=normalized_user_msg)
                if raw is None or "UNSURE_FROM_DATA" in raw:
                    answer_lines.append(DEFAULT_WARRANTY_TEXT)
                else:
                    answer_lines.append(raw.strip())
            else:
                answer_lines.append(DEFAULT_WARRANTY_TEXT)
        if any(k in low_both for k in gst_keywords):
            if found_gst:
                # again prefer dataset context, but if not explicit use default
                # quick heuristic: extract line(s) with "gst" or "%"
                gst_matches = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]*?(gst|%)[^|\n]*)", "\n\n".join(docs), flags=re.IGNORECASE)
                if gst_matches:
                    gst_vals = [m[1].strip() for m in gst_matches]
                    answer_lines.append("GST information found: " + "; ".join(sorted(set(gst_vals))))
                else:
                    answer_lines.append(DEFAULT_GST_TEXT)
            else:
                answer_lines.append(DEFAULT_GST_TEXT)

        return ChatResponse(answer="\n\n".join(answer_lines), used_context=docs, from_fallback=False)

    # If not matched earlier, proceed to normal embedding and RAG
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

    docs.extend(_query_collection(end_customer_collection, "END_CUSTOMER", query_vec))
    docs.extend(_query_collection(spare_parts_collection, "SPARE_PARTS", query_vec))
    docs.extend(_query_collection(dealer_collection, "DEALERS", query_vec))

    if not docs:
        print("[CHAT] No relevant documents found in any index.")
        return ChatResponse(
            answer=fallback(),
            used_context=[],
            from_fallback=True,
        )

    unique_docs = list(dict.fromkeys(docs))
    context = "\n\n---\n\n".join(unique_docs)

    # brand-only helper (hardened to avoid greetings/stopwords and generic terms like 'transport')
    def is_brand_only(text: str) -> bool:
        text = (text or "").strip()
        if not text:
            return False
        # Avoid treating simple stopwords or short replies as brand queries
        stopwords = {"hi", "hello", "hey", "yes", "no", "ok", "okay", "thanks", "thank", "please"}
        lowt = text.lower()
        if lowt in stopwords:
            return False
        # Generic nouns/phrases that should NOT be treated as brand names
        generic_blocklist = {
            "transport", "transportation", "transport details", "transportation details", "transport cost", "transportation cost",
            "transport how", "transportation how", "shipping", "shipping details", "delivery", "delivery details",
            "transportation?", "transport?", "how to transport", "how transport"
        }
        # If text contains any blocklisted generic term, it's not a brand-only query
        for g in generic_blocklist:
            if g in lowt:
                return False
        if re.search(r"\d", text):
            return False
        if len(text.split()) > 3:
            return False
        # If text is extremely short (1-2 chars) avoid brand-only
        if len(text) <= 2:
            return False
        # Also ensure the text looks like a brand token (contains letters)
        if not re.search(r"[A-Za-z]", text):
            return False
        return True

    # If brand-only detection should consider translated English (so local-language brand names still work),
    # we will test both original and translated.
    if is_brand_only(normalized_user_msg) or (low_en and is_brand_only(low_en)):
        brand = normalized_user_msg.strip()
        candidate_models: List[str] = []
        for d in unique_docs:
            pairs = re.findall(r"([A-Za-z0-9 _/()%-]+):\s*([^|\n]+)", d)
            for (k, v) in pairs:
                k_clean = k.strip().lower()
                val = v.strip()
                if "model" in k_clean or "machine" in k_clean or "machine model" in k_clean:
                    if re.search(r"[A-Za-z]+\d", val) or re.search(r"\d+[A-Za-z]+", val) or "-" in val or brand.lower() in val.lower():
                        candidate_models.append(val)
        if not candidate_models:
            for d in unique_docs:
                parts = re.split(r"[|,/\\\n]+", d)
                for p in parts:
                    token = p.strip()
                    if not token:
                        continue
                    if re.search(r"[A-Za-z]+\d", token) or re.search(r"\d+[A-Za-z]+", token) or "-" in token:
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

    # model (VJ) specific handling: return full rows for the model followed by a bullet list of all compatible machines
    vj_match_user = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", normalized_user_msg, re.IGNORECASE)
    if vj_match_user:
        # build normalized base token (e.g., 'vj20')
        parts = [vj_match_user.group(1) or "", vj_match_user.group(2) or ""]
        # user may have typed 'VJ20 HD' — capture that but we'll try matching logic below
        if vj_match_user.group(3):
            parts.append(vj_match_user.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())

        # Candidate search order:
        # 1) exact token user asked (key_norm)
        # 2) if no exact match and user didn't ask HD, try key_norm + 'hd'
        # 3) if still none, try rows that contain key_norm anywhere (broader)
        # 4) fallback to collecting rows via the compatibility collector
        candidate_keys = [key_norm]
        if not key_norm.endswith("hd"):
            candidate_keys.append(key_norm + "hd")
        else:
            # if user asked VJXXHD explicitly, try without hd too
            candidate_keys.append(re.sub(r"hd$", "", key_norm))

        src_rows: List[str] = []
        matched_key: Optional[str] = None

        for cand in candidate_keys:
            cand_low = cand.lower()
            found = []
            for d in unique_docs:
                if cand_low in d.lower():
                    found.append(d)
            if found:
                src_rows = found
                matched_key = cand_low
                break

        # If still no rows found, do a broader scan: any doc containing the base numeric token may match
        if not src_rows:
            broader = []
            for d in unique_docs:
                if key_norm in d.lower():
                    broader.append(d)
            if broader:
                src_rows = broader
                matched_key = key_norm

        # If still empty, use the compatibility collector as last resort
        if not src_rows:
            compat_set, src_rows2 = _collect_all_compatible_machines_or_breakers(key_norm, unique_docs)
            if src_rows2:
                src_rows = src_rows2
                # if compat_set contains a vj variant, pick the first vj variant as matched key (prefer hd if present)
                chosen = None
                vj_variants = sorted([v for v in compat_set if re.fullmatch(r"vj\d+hd?", v)])
                if vj_variants:
                    # prefer HD variant if exists
                    for vv in vj_variants:
                        if vv.endswith("hd"):
                            chosen = vv
                            break
                    if not chosen:
                        chosen = vj_variants[0]
                matched_key = chosen or key_norm

        if not src_rows:
            # nothing found at all
            return ChatResponse(answer=fallback(), used_context=unique_docs, from_fallback=True)

        # If matched_key is None, set default
        if not matched_key:
            matched_key = key_norm

        # Determine display name (e.g., "VJ20 HD" or "VJ20")
        m_m = re.match(r"vj(\d{1,3})(hd)?", matched_key, flags=re.IGNORECASE)
        display_model = key_raw.upper()
        if m_m:
            num = m_m.group(1)
            hd_flag = m_m.group(2)
            display_model = f"VJ{num}" + (f" HD" if hd_flag else "")

        # Merge all fields across rows to produce ONE consolidated set of key: value pairs for the breaker
        merged_fields = _merge_rows_to_field_map(src_rows)

        # ---------------------------
        # FILTER: fields to hide for model-only responses
        hide_fields_when_model_only = {
            "sku", "stock available", "stock on hand", "warehouse name", "warehouse", "selling price- end customer",
            "selling price", "selling price end customer", "selling price - end customer", "stock", "part brand", "part", "item name",
            "spare parts", "spare part", "spare parts sku", "sparepart", "part sku", "warehouse name",
            # synonyms-related keys to hide
            "synonmys of item name", "synonym", "synonyms", "synonyms of item name", "synonyms of itemname", "synonyms of item", "synonym of item name"
        }
        # ---------------------------

        # Detect if the user requested a specific detail (priority)
        detail_map = {
            "price": ["price", "mrp", "price incl", "price incl gst", "price including gst", "price including", "price without gst", "price including gst", "price without gst", "minimum price"],
            "sku": ["sku"],
            "stock": ["stock on hand", "stock", "stock available"],
            "warranty": ["warranty", "warranty period", "warranty terms"],
            "gst": ["gst", "%"],
            "chisel": ["chisel", "chisel dia", "chisel dia ( mm )", "chisel dia ( mm )", "chisel dia (mm)"],
            "weight": ["weight", "breaker weight", "breaker weight ( kg )", "breaker weight ( kg )"],
            "impact": ["impact", "impact in joules", "impact rate ( bpm )", "impact rate"],
            "pin": ["pin size", "pin size ( mm )"],
            "primary application": ["primary application", "primary application:"],
            "machine": ["machine model", "machine", "machine brand"],
        }
        # find which detail user explicitly asked for by scanning the raw (lowercased) user message
        requested_detail_key = None
        for dk in detail_map.keys():
            if dk in low_both:
                requested_detail_key = dk
                break

        # helper to find a value in merged_fields by candidate fragments
        def _find_field_value_by_candidates(cands: List[str], field_map: Dict[str, str]) -> Optional[str]:
            cand_lower = [c.lower() for c in cands]
            # prefer exact substring matches in key names
            for k in field_map:
                lk = k.lower()
                for cand in cand_lower:
                    if cand in lk:
                        val = field_map.get(k)
                        if val:
                            return val
            # fallback: try to find any key where value contains typical signal for candidate (rare)
            return None

        # If user requested a specific detail, try to obtain its value (prefer merged_fields; for price also search docs)
        requested_detail_value = None
        requested_label = None
        if requested_detail_key:
            candidates = detail_map.get(requested_detail_key, [])
            # search merged fields
            v = _find_field_value_by_candidates(candidates, merged_fields)
            if v:
                requested_detail_value = v
                requested_label = candidates[0].title()
            else:
                # if 'price' requested, try _find_price_for_model across docs
                if requested_detail_key == "price":
                    # try to query docs for matched_key model
                    try:
                        query_vec = embed_query(matched_key)
                        docs_for_price: List[str] = []
                        docs_for_price.extend(_query_collection(end_customer_collection, "END_CUSTOMER", query_vec))
                        docs_for_price.extend(_query_collection(spare_parts_collection, "SPARE_PARTS", query_vec))
                        docs_for_price.extend(_query_collection(dealer_collection, "DEALERS", query_vec))
                        price_found = _find_price_for_model(matched_key, docs_for_price)
                        if price_found:
                            requested_detail_value = price_found
                            requested_label = "Price"
                        else:
                            requested_detail_value = None
                            requested_label = "Price"
                    except Exception as e:
                        print(f"[ERROR] Failed to embed/query for price priority: {e}")
                        requested_detail_value = None
                        requested_label = "Price"
                else:
                    # not price and not in merged_fields
                    requested_detail_value = None
                    requested_label = requested_detail_key.title()

        # Format details as bullet points (one block only), but put requested detail first if found / attempted
        details_lines = [f"**Details for {display_model}:**"]

        # If requested detail exists, put it first
        if requested_detail_key:
            # If we have a value, show it; otherwise say not found in dataset for explicitness
            if requested_detail_value:
                details_lines.append(f"- {requested_label}: {requested_detail_value}")
            else:
                details_lines.append(f"- {requested_label}: Not found in dataset.")

        # Build the rest of the details (excluding hidden fields and excluding field already shown)
        preferred_order = ["model", "breaker model", "sku", "brand", "machine brand", "capacity", "tonnage", "price", "warranty"]
        used_keys = set()
        # If requested detail came from a merged_fields key, exclude that key when listing remaining fields
        if requested_detail_key and requested_detail_value:
            for k, v in merged_fields.items():
                if v == requested_detail_value:
                    used_keys.add(k)

        for k in preferred_order:
            for key in list(merged_fields.keys()):
                if k == key:
                    val = merged_fields.get(key)
                    if val:
                        # skip hidden keys
                        if key.lower() in hide_fields_when_model_only:
                            continue
                        # skip if already used
                        if key in used_keys:
                            continue
                        details_lines.append(f"- {key.title()}: {val}")
                        used_keys.add(key)
        # remaining keys (exclude hidden ones)
        for key in sorted(merged_fields.keys()):
            if key in used_keys:
                continue
            if key.lower() in hide_fields_when_model_only:
                continue
            val = merged_fields.get(key)
            if val:
                details_lines.append(f"- {key.title()}: {val}")

        # Extract compatible machines (Brand - Model). Use dedicated parser to avoid listing breakers.
        model_keys = {matched_key, key_norm}
        compatible_machines = _parse_compatible_machine_tokens_from_rows(src_rows, all_docs=unique_docs, model_keys=model_keys)

        # If parser gives nothing, also try scanning all unique_docs for any 'machine model'/'machine brand' rows that mention this breaker
        if not compatible_machines:
            fallback_rows = []
            for d in unique_docs:
                if matched_key in d.lower() or key_norm in d.lower():
                    fallback_rows.append(d)
            compatible_machines = _parse_compatible_machine_tokens_from_rows(fallback_rows, all_docs=unique_docs, model_keys=model_keys)

        compat_lines = ["\n**Compatible machines:**"]
        if compatible_machines:
            for cm in compatible_machines:
                compat_lines.append(f"- {cm}")
        else:
            compat_lines.append("No compatible machines listed in dataset.")

        # Only add followup if below limit. Also add hidden machine-readable comment for followups (CLIENT will use to resolve 'price'/'brochure' replies).
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

        # Attach a hidden HTML comment that the front-end ignores but we can parse on subsequent requests
        hidden_comment = f"<!--MODEL_CTX:{matched_key}|OPTIONS:brochure,connect,price,specs-->"

        reply_text = "\n\n".join(["\n".join(details_lines), "\n".join(compat_lines)]) + followup_text + "\n\n" + hidden_comment

        # find brochure for matched_key (try matched_key then alt without/with hd)
        fname = BROCHURE_MAP.get(matched_key)
        if not fname:
            alt = re.sub(r"hd$", "", matched_key)
            fname = BROCHURE_MAP.get(alt)
            if not fname and not matched_key.endswith("hd"):
                fname = BROCHURE_MAP.get(matched_key + "hd")

        brochure_url = None
        if fname:
            base = str(request.base_url).rstrip("/")
            brochure_url = f"{base}/brochures/view/{fname}"

        return ChatResponse(answer=reply_text, used_context=src_rows or unique_docs, from_fallback=False, brochure_url=brochure_url, followup_meta={"model": matched_key, "options": ["brochure","connect","price","specs"]})

    # tonnage handling
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
            # Hidden comment does not have a model key in this flow (generic), but include options
            hidden_comment = "<!--MODEL_CTX:|OPTIONS:brochure,connect,price,specs-->"
            return ChatResponse(answer="\n".join(lines) + followup_text + "\n\n" + hidden_comment, used_context=matched_rows or unique_docs, from_fallback=False)

    # Fallback to LLM with context (RAG)
    raw = generate_with_groq(context=context, user_question=normalized_user_msg, user_ton=_extract_ton_from_text(normalized_user_msg))
    if raw is None:
        return ChatResponse(answer=fallback(), used_context=unique_docs, from_fallback=True)

    raw = raw.strip()

    if "UNSURE_FROM_DATA" in raw:
        return ChatResponse(answer=fallback(), used_context=unique_docs, from_fallback=True)

    # try to detect if LLM output contains a VJ model; if so find brochure and add followup hidden comment
    brochure_url = None
    m = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd|hd$))?\b", normalized_user_msg, re.IGNORECASE)
    model_key_norm = None
    if m:
        parts = [m.group(1) or "", m.group(2) or ""]
        if m.group(3):
            parts.append(m.group(3))
        key_raw = "".join(parts)
        key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
        model_key_norm = key_norm
        fname = BROCHURE_MAP.get(key_norm)
        if not fname:
            alt_key = re.sub(r"hd$", "", key_norm)
            fname = BROCHURE_MAP.get(alt_key)
        if fname:
            base = str(request.base_url).rstrip("/")
            brochure_url = f"{base}/brochures/view/{fname}"

    # Also check if the model is referenced in the raw LLM output (if no explicit mention by user)
    if not brochure_url:
        m2 = re.search(r"\b(vj)[\s\-]*?(\d{1,3})(?:\s*(hd))?\b", raw, re.IGNORECASE)
        if m2:
            key_raw = "".join([m2.group(1) or "", m2.group(2) or ""] + ([m2.group(3)] if m2.group(3) else []))
            key_norm = re.sub(r"[^a-z0-9]", "", key_raw.lower())
            fname = BROCHURE_MAP.get(key_norm)
            if fname:
                base = str(request.base_url).rstrip("/")
                brochure_url = f"{base}/brochures/view/{fname}"
                model_key_norm = key_norm

    final_answer = raw

    # If we plan followups, add hidden comment to keep model context for the client's next reply
    followups_text = ""
    hidden_comment = ""
    # Only add followups when the raw (LLM) response seems to mention a specific model or we otherwise detected a model
    if prior_followups < SALES_NUDGE_LIMIT and (model_key_norm or re.search(r"\b(vj)\s*\d{1,3}", raw, re.IGNORECASE)):
        followups_text = (
            "\n\n**Would you like any of the following for this model?**\n"
            "- Brochure (reply 'brochure')\n"
            "- Connect me with a dealer/contact for a quote (reply 'connect')\n"
            "- Get a price / proforma (reply 'price')\n"
            "- More specs or warranty details (reply 'specs' or 'warranty')\n\n"
            "Reply with which option you'd like, or 'no' to skip."
        )
        if model_key_norm:
            hidden_comment = f"<!--MODEL_CTX:{model_key_norm}|OPTIONS:brochure,connect,price,specs-->"
        else:
            # no explicit model key, keep generic hidden comment (client will handle)
            hidden_comment = "<!--MODEL_CTX:|OPTIONS:brochure,connect,price,specs-->"

    if followups_text:
        final_answer = f"{final_answer}{followups_text}"
    if hidden_comment:
        final_answer = f"{final_answer}\n\n{hidden_comment}"

    # If brochure_url found, also return it as a separate field (frontend will avoid duplicate rendering)
    return ChatResponse(answer=final_answer, used_context=unique_docs, from_fallback=False, brochure_url=brochure_url)

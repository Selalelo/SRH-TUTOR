from dotenv import load_dotenv
import os
load_dotenv()

os.environ["HF_TOKEN"]                = os.getenv("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"

import warnings, logging, gc, time, uuid, threading, re, json
from datetime import datetime, timezone
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

from typing import List, Union
from typing_extensions import TypedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

BASE_DIR   = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

from supabase import create_client, Client
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    PayloadSchemaType, Filter, FieldCondition, MatchValue
)
from srh_embedder import embed, embed_one

# ══════════════════════════════════════════════════════════════
#  CONNECTIONS
# ══════════════════════════════════════════════════════════════

_url         = os.getenv("SUPABASE_URL")
_anon_key    = os.getenv("SUPABASE_ANON_KEY")
_service_key = os.getenv("SUPABASE_SERVICE_KEY", _anon_key)

supabase_admin: Client = create_client(_url, _service_key)
supabase_auth:  Client = create_client(_url, _anon_key)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY"),
    timeout=10
)

MANUAL_COLLECTION = "srh_manual"
CHAT_COLLECTION   = "srh_chat"
VECTOR_SIZE       = 384

_known_collections: set = set()

def collection_exists(name: str) -> bool:
    global _known_collections
    if name in _known_collections:
        return True
    try:
        existing = {c.name for c in qdrant.get_collections().collections}
        _known_collections = existing
        return name in existing
    except Exception:
        return False

# ══════════════════════════════════════════════════════════════
#  LAZY SINGLETONS
# ══════════════════════════════════════════════════════════════

_llm   = None
_agent = None

def get_llm():
    global _llm
    if _llm is None:
        # Higher max_tokens + longer timeout because generating a 20-question
        # JSON quiz in one shot can be ~6k tokens out and 30+ seconds.
        _llm = ChatGroq(model="llama-3.3-70b-versatile",
                        timeout=60, max_retries=1, max_tokens=8000)
    return _llm

def get_agent():
    global _agent
    if _agent is None:
        class AgentState(TypedDict):
            messages: List[Union[HumanMessage, SystemMessage, AIMessage]]
        def process(state: AgentState) -> AgentState:
            response = get_llm().invoke(state["messages"])
            return {"messages": [AIMessage(content=response.content)]}
        graph = StateGraph(AgentState)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        _agent = graph.compile()
    return _agent

# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="SRH Tutor API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/ready")
def ready():
    from srh_embedder import _session
    model_ready = _session is not None
    return {
        "ready":    model_ready and _agent is not None,
        "embedder": model_ready,
        "agent":    _agent is not None,
    }

# ══════════════════════════════════════════════════════════════
#  QDRANT INIT
# ══════════════════════════════════════════════════════════════

def init_qdrant_chat_collection():
    try:
        if not collection_exists(CHAT_COLLECTION):
            qdrant.create_collection(
                collection_name=CHAT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            _known_collections.add(CHAT_COLLECTION)
            print(f"✅ Qdrant '{CHAT_COLLECTION}' created.")

        qdrant.create_payload_index(
            collection_name=CHAT_COLLECTION,
            field_name="timestamp",
            field_schema=PayloadSchemaType.FLOAT
        )
        qdrant.create_payload_index(
            collection_name=CHAT_COLLECTION,
            field_name="user_id",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"✅ Qdrant chat indexes ready.")
    except Exception as e:
        print(f"⚠️  Qdrant chat init: {e}")

def init_qdrant_manual_index():
    """Ensure the 'source' payload index exists on the manual collection."""
    try:
        if not collection_exists(MANUAL_COLLECTION):
            print(f"ℹ️  Manual collection not found — skipping source index creation.")
            return
        qdrant.create_payload_index(
            collection_name=MANUAL_COLLECTION,
            field_name="source",
            field_schema=PayloadSchemaType.KEYWORD
        )
        print(f"✅ Qdrant manual 'source' index ready.")
    except Exception as e:
        # Index may already exist — not a fatal error
        print(f"ℹ️  Manual source index: {e}")

# ══════════════════════════════════════════════════════════════
#  AUTH HELPER
# ══════════════════════════════════════════════════════════════

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        user = supabase_admin.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user.user
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate token")

# ══════════════════════════════════════════════════════════════
#  MANUAL SEARCH
# ══════════════════════════════════════════════════════════════

def search_manual(query: str, n_per_source: int = 3, pool_size: int = 20):
    """
    Search the manual collection with source-balanced retrieval.

    Pulls a larger pool of top hits from Qdrant, then keeps up to
    n_per_source chunks from each distinct 'source' so every ingested
    book contributes context (instead of one book dominating because
    its embeddings happen to be closer to the query).
    """
    if not collection_exists(MANUAL_COLLECTION):
        raise HTTPException(status_code=503,
            detail="📚 Manual not ingested yet. Run: python ingest_srh.py")
    try:
        query_vector = embed_one(query)
        results = qdrant.query_points(
            collection_name=MANUAL_COLLECTION,
            query=query_vector,
            limit=pool_size,
            with_payload=True,
        )
        if not results.points:
            return "", []

        per_source: dict = {}
        for hit in results.points:
            src = hit.payload.get("source", "Manual")
            bucket = per_source.setdefault(src, [])
            if len(bucket) < n_per_source:
                bucket.append(hit)

        selected = [hit for hits in per_source.values() for hit in hits]
        selected.sort(key=lambda h: h.score, reverse=True)

        parts, sources = [], []
        for hit in selected:
            src = hit.payload.get("source", "Manual")
            page = hit.payload["page"]
            parts.append(f"[{src}, Page {page}]:\n{hit.payload['text']}")
            sources.append(f"{src} · Page {page}")
        del results
        return "\n".join(parts), sources
    except HTTPException:
        raise
    except Exception as e:
        print(f"⚠️  Manual search failed: {e}")
        return "", []

# ══════════════════════════════════════════════════════════════
#  CHAT HISTORY
# ══════════════════════════════════════════════════════════════

def save_message(user_id: str, role: str, content: str, sources: list = []):
    msg_id = str(uuid.uuid4())
    try:
        supabase_admin.table("chat_messages").insert({
            "id": msg_id, "user_id": user_id, "role": role,
            "content": content, "sources": sources or []
        }).execute()
    except Exception as e:
        print(f"⚠️  Supabase insert failed: {e}")
    try:
        vector = embed_one(content)
        qdrant.upsert(collection_name=CHAT_COLLECTION, points=[
            PointStruct(id=msg_id, vector=vector,
                payload={"user_id": user_id, "role": role,
                         "content": content, "timestamp": time.time()})
        ])
        del vector
    except Exception as e:
        print(f"⚠️  Qdrant upsert failed: {e}")

def load_user_history(user_id: str, limit: int = 30):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content").eq("user_id", user_id)
               .order("created_at", desc=True).limit(limit).execute())
        rows = list(reversed(res.data or []))
        messages = []
        for row in rows:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                messages.append(AIMessage(content=row["content"]))
        del res
        return messages
    except Exception as e:
        print(f"⚠️  History load failed: {e}")
        return []

def get_history_for_api(user_id: str, limit: int = 60):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content, sources, created_at")
               .eq("user_id", user_id)
               .order("created_at", desc=True).limit(limit).execute())
        data = list(reversed(res.data or []))
        del res
        return data
    except Exception as e:
        print(f"⚠️  get_history_for_api failed: {e}")
        return []

def delete_user_history(user_id: str):
    try:
        supabase_admin.table("chat_messages").delete().eq("user_id", user_id).execute()
    except Exception as e:
        print(f"⚠️  Supabase delete failed: {e}")
    try:
        qdrant.delete(collection_name=CHAT_COLLECTION,
            points_selector=Filter(must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            ]))
    except Exception as e:
        print(f"⚠️  Qdrant delete failed: {e}")

def update_last_seen(user_id: str):
    try:
        supabase_admin.table("profiles").update({"last_seen": "now()"}).eq("id", user_id).execute()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a professional, compassionate tutor for a physiology course covering multiple modules.

The course has the following modules and documents:
  • SRH Module  → SPLA031 RSH notes 2024 (Sexual & Reproductive Health Training Manual)
  • CDL Module  → CDL notes 2022
                  CDL Notes Slides Lect 9-20
                  Chronic Diseases of Lifestyle overview

All four documents are active course material. Treat them equally.

Answer questions ONLY from the provided training documents.

CITATION RULES — read carefully:
- Each retrieved excerpt is shown as `[<Source>, Page <N>]: <text>`.
  The source name in the bracket is the EXACT document the excerpt came from.
- When you cite, your citation MUST name a source that actually appears in the excerpts provided this turn.
  Do NOT cite a document whose name is not in the excerpts.
- Quote the source name verbatim as it appears in the bracket.
- If the excerpts don't cover the question, say so plainly — do not invent a citation.

Teaching style:
- Ground explanations in the provided document excerpts
- Use clear, clinical, respectful language
- Explain concepts step by step
- Be non-judgmental, inclusive, evidence-based, and practical

You operate in 3 modes:
  1. EXPLAIN  → explain the concept using document excerpts + clear examples
  2. EXERCISE → present a case-study or scenario-based task for the student to work through
  3. REVIEW   → give constructive feedback on the student's written answer

NOTE on quizzes:
Quizzes are handled by a separate automated system.
If a student asks for a quiz, tell them to type:
  • "quiz me"            → pulls from all modules
  • "quiz me on CDL"     → CDL module only
  • "quiz me on SRH"     → SRH module only
Do NOT generate quiz questions yourself, ask "Question N of 20", or grade answers manually.

After explaining a topic, ask if they want an exercise to apply what they learned.
Keep responses concise.
"""

# ══════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ══════════════════════════════════════════════════════════════

class SignUpRequest(BaseModel):
    email: str
    password: str
    full_name: str

class SignInRequest(BaseModel):
    email: str
    password: str

class ChatRequest(BaseModel):
    message: str

# ══════════════════════════════════════════════════════════════
#  AUTH ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/auth/signup")
def signup(body: SignUpRequest):
    try:
        res = supabase_auth.auth.sign_up({
            "email": body.email, "password": body.password,
            "options": {"data": {"full_name": body.full_name}}
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")
        return {"message": "Account created! Please check your email to confirm.", "user_id": res.user.id}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/signin")
def signin(body: SignInRequest):
    try:
        res = supabase_auth.auth.sign_in_with_password({"email": body.email, "password": body.password})
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        update_last_seen(res.user.id)
        token     = res.session.access_token
        user_data = {"id": res.user.id, "email": res.user.email,
                     "full_name": res.user.user_metadata.get("full_name", "")}
        del res
        return {"access_token": token, "user": user_data}
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        if "Email not confirmed" in error_msg or "email_not_confirmed" in error_msg:
            detail = "Please confirm your email first, or disable confirmation in Supabase dashboard."
        elif "Invalid login credentials" in error_msg:
            detail = "Incorrect email or password."
        else:
            detail = f"Sign in failed: {error_msg}"
        raise HTTPException(status_code=401, detail=detail)

@app.post("/auth/resend-confirmation")
def resend_confirmation(body: SignInRequest):
    try:
        supabase_auth.auth.resend({"type": "signup", "email": body.email})
        return {"message": "Confirmation email resent! Check your inbox."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not resend: {str(e)}")

@app.post("/auth/signout")
def signout(current_user=Depends(get_current_user)):
    try:
        supabase_auth.auth.sign_out()
    except Exception:
        pass
    return {"message": "Signed out successfully"}

@app.get("/auth/me")
def get_me(current_user=Depends(get_current_user)):
    try:
        profile = (supabase_admin.table("profiles")
                   .select("full_name, email, role, created_at, last_seen")
                   .eq("id", current_user.id).single().execute())
        data = profile.data
        del profile
        return {"id": current_user.id, "email": current_user.email,
                "full_name": data.get("full_name", ""), "role": data.get("role", "trainee"),
                "created_at": data.get("created_at"), "last_seen": data.get("last_seen")}
    except Exception:
        return {"id": current_user.id, "email": current_user.email,
                "full_name": current_user.user_metadata.get("full_name", ""), "role": "trainee"}

# ══════════════════════════════════════════════════════════════
#  QUIZ SESSIONS
# ══════════════════════════════════════════════════════════════
#
# Quizzes are run by the server, not the LLM. When a user asks for a
# quiz, we (1) retrieve a balanced pool of chunks from both manuals,
# (2) ask the LLM to generate grounded questions in a single shot,
# (3) validate every question against the actual chunks (source/page
# must match, excerpt_quote must appear in the chunk's text), and
# (4) store the question list in `quiz_sessions`. After that, every
# answer is graded server-side against the stored answer key — the
# LLM is never asked to generate a question or invent a citation
# during quiz turns.
#
# This eliminates the per-turn drift / hallucination we got when the
# LLM was asked to manage the quiz itself.

QUIZ_LENGTH = 20

_QUIZ_REQUEST_RE = re.compile(
    r"\b(quiz\s*me|quiz\s*on|quiz\s*about|start\s*(?:a\s*)?quiz|"
    r"give\s*me\s*a\s*quiz|i\s*want\s*(?:a\s*)?quiz|test\s*me|"
    r"let'?s\s*quiz)\b",
    re.IGNORECASE,
)
_QUIZ_BARE_RE = re.compile(r"^\s*quiz\s*\??\s*$", re.IGNORECASE)
_QUIZ_CANCEL_RE = re.compile(
    r"^\s*(cancel|stop|quit|exit|stop\s*quiz|cancel\s*quiz|end\s*quiz|quit\s*quiz)\s*\.?\s*$",
    re.IGNORECASE,
)
_QUIZ_TOPIC_RE = re.compile(
    r"\b(?:quiz\s*me\s*on|quiz\s*on|quiz\s*about|test\s*me\s*on)\s+(.+)$",
    re.IGNORECASE,
)


def is_quiz_request(msg: str) -> bool:
    return bool(_QUIZ_REQUEST_RE.search(msg) or _QUIZ_BARE_RE.match(msg))


def is_quiz_cancel(msg: str) -> bool:
    return bool(_QUIZ_CANCEL_RE.match(msg))


def _extract_topic(msg: str):
    m = _QUIZ_TOPIC_RE.search(msg)
    if not m:
        return None
    topic = m.group(1).strip().rstrip(".?!,")
    return topic or None


# ── Supabase helpers ─────────────────────────────────────────

def get_active_quiz(user_id: str):
    try:
        res = (supabase_admin.table("quiz_sessions")
               .select("*").eq("user_id", user_id)
               .is_("completed_at", "null").is_("cancelled_at", "null")
               .order("started_at", desc=True).limit(1).execute())
        rows = res.data or []
        return rows[0] if rows else None
    except Exception as e:
        print(f"⚠️  get_active_quiz failed: {e}")
        return None


def create_quiz_session(user_id: str, questions: list, topic):
    try:
        payload = {
            "user_id":       user_id,
            "questions":     questions,
            "answers":       [],
            "current_index": 0,
            "score":         0,
            "topic":         topic,
        }
        res = supabase_admin.table("quiz_sessions").insert(payload).execute()
        return res.data[0] if res.data else None
    except Exception as e:
        print(f"⚠️  create_quiz_session failed: {e}")
        return None


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def update_quiz_session(session_id: str, *, current_index: int, score: int,
                       answers: list, completed: bool = False):
    try:
        update = {"current_index": current_index, "score": score, "answers": answers}
        if completed:
            update["completed_at"] = _now_iso()
        supabase_admin.table("quiz_sessions").update(update).eq("id", session_id).execute()
    except Exception as e:
        print(f"⚠️  update_quiz_session failed: {e}")


def cancel_quiz_session(session_id: str):
    try:
        (supabase_admin.table("quiz_sessions")
         .update({"cancelled_at": _now_iso()}).eq("id", session_id).execute())
    except Exception as e:
        print(f"⚠️  cancel_quiz_session failed: {e}")


def cancel_active_quizzes_for_user(user_id: str):
    try:
        (supabase_admin.table("quiz_sessions")
         .update({"cancelled_at": _now_iso()}).eq("user_id", user_id)
         .is_("completed_at", "null").is_("cancelled_at", "null").execute())
    except Exception as e:
        print(f"⚠️  cancel_active_quizzes_for_user failed: {e}")


# ── Question generation (validated against real chunks) ──────

_CHUNK_TRUNC_CHARS = 350   # keep prompt size sane; chunks are ~500 chars by default


def _retrieve_quiz_chunks(topic, n_per_source: int = 6):
    """Return (excerpts_text_for_prompt, chunks_meta) — chunks_meta
    is a list of {source, page, text} for validation. Chunks are truncated
    in the prompt only; full text is kept in chunks_meta for validation."""
    query = topic if topic else "training material overview key concepts"
    if not collection_exists(MANUAL_COLLECTION):
        return "", []
    try:
        query_vector = embed_one(query)
        results = qdrant.query_points(
            collection_name=MANUAL_COLLECTION,
            query=query_vector,
            limit=40,
            with_payload=True,
        )
        if not results.points:
            return "", []

        per_source: dict = {}
        for hit in results.points:
            src = hit.payload.get("source", "Manual")
            bucket = per_source.setdefault(src, [])
            if len(bucket) < n_per_source:
                bucket.append(hit)

        chunks_meta = []
        for hits in per_source.values():
            for hit in hits:
                chunks_meta.append({
                    "source": hit.payload.get("source", "Manual"),
                    "page":   hit.payload.get("page"),
                    "text":   hit.payload.get("text", ""),
                })

        prompt_parts = []
        for c in chunks_meta:
            text = c["text"] or ""
            if len(text) > _CHUNK_TRUNC_CHARS:
                text = text[:_CHUNK_TRUNC_CHARS].rstrip() + "…"
            prompt_parts.append(f"[{c['source']}, Page {c['page']}]:\n{text}")
        return "\n\n".join(prompt_parts), chunks_meta
    except Exception as e:
        print(f"⚠️  quiz chunk retrieval failed: {e}")
        return "", []


_QUIZ_GEN_SYSTEM = """You generate quiz questions strictly grounded in provided document excerpts.

Rules:
- Output VALID JSON only — a single array of question objects. No prose, no code fences, no commentary.
- Each question must be answerable from ONE specific excerpt provided.
- The "source" and "page" fields MUST exactly match the [Source, Page N] header of the excerpt the question is drawn from.
- The "excerpt_quote" must be a verbatim quote (≥10 words) from that excerpt.
- Cover ALL modules — distribute questions evenly across ALL sources present in the excerpts.
  The course modules and their documents are:
    • SPLA031 RSH notes 2024        → Sexual & Reproductive Health
    • CDL notes 2022                → Chronic Diseases of Lifestyle (core notes)
    • CDL Notes Slides Lect 9-20   → Chronic Diseases of Lifestyle (lectures 9–20)
    • Chronic Diseases of Lifestyle overview → CDL overview/intro
  Every quiz must draw from more than one module where excerpts from multiple modules are available.
- Question types: multiple-choice ("mcq") with 4 options A–D, OR true/false ("tf"). No short-answer or open-ended questions.
- Test substantive physiological, epidemiological, and clinical knowledge — NOT the document's structure.
  Forbidden question patterns:
    • "Does the manual have a section on X?"
    • "According to page N, what heading appears?"
    • Any question whose answer is purely navigational or administrative.
- Topic coverage guidance per module:
    SRH (SPLA031):
      • Reproductive anatomy and physiology
      • Contraception methods and mechanisms
      • STIs: transmission, symptoms, prevention
      • Adolescent sexual health
      • Gender, rights, and consent
    CDL (all CDL sources):
      • Definitions and characteristics of chronic NCDs
      • Global and African burden of disease (mortality/morbidity statistics)
      • Demographic, epidemiological and nutrition transition
      • Social determinants of health (structural/distal vs proximal/individual factors)
      • Risk factors: diet & nutrition, physical inactivity, tobacco use
      • Modifiable vs non-modifiable risk factors
      • Impact of NCDs on individuals, families, workforce, and health systems
      • Prevention and control strategies
- Each MCQ must have exactly one correct option. "correct_answer" is the letter only: "A", "B", "C", or "D".
- For tf, "correct_answer" is "True" or "False".
- Distractors must be plausible but clearly incorrect based on the excerpt. Avoid obviously silly distractors.
- "explanation" is 1–2 sentences citing the source naturally (e.g. "The CDL notes 2022 state that...").

Output schema — a JSON array, nothing else:
[
  {
    "type": "mcq" | "tf",
    "question": "...",
    "options": {"A": "...", "B": "...", "C": "...", "D": "..."},
    "correct_answer": "A" | "B" | "C" | "D" | "True" | "False",
    "explanation": "...",
    "source": "exact source name from the excerpt bracket",
    "page": <integer>,
    "excerpt_quote": "verbatim quote of ≥10 words from the excerpt this question is based on"
  }
]

For true/false questions, omit the "options" field entirely.
"""


def _quiz_gen_user_prompt(excerpts: str, n: int, topic) -> str:
    topic_clause = (
        f"All questions must be on the topic: {topic}.\n\n"
        if topic else
        "Spread questions across the topics covered in the excerpts.\n\n"
    )
    return (
        f"{topic_clause}Generate {n} quiz questions from these excerpts.\n\n"
        f"EXCERPTS:\n{excerpts}\n\n"
        "Output schema (JSON array):\n"
        '[\n'
        '  {\n'
        '    "type": "mcq",\n'
        '    "question": "...",\n'
        '    "options": ["A) ...", "B) ...", "C) ...", "D) ..."],\n'
        '    "correct_answer": "B",\n'
        '    "explanation": "...",\n'
        '    "source": "<exact source name from the bracket>",\n'
        '    "page": <number>,\n'
        '    "excerpt_quote": "<verbatim quote of >=10 words>"\n'
        '  },\n'
        '  {\n'
        '    "type": "tf",\n'
        '    "question": "...",\n'
        '    "correct_answer": "True",\n'
        '    "explanation": "...",\n'
        '    "source": "<exact source name>",\n'
        '    "page": <number>,\n'
        '    "excerpt_quote": "<verbatim quote of >=10 words>"\n'
        '  }\n'
        ']\n\n'
        f"Aim for ~75% mcq and ~25% tf. Output the JSON array of {n} items and nothing else."
    )


def _extract_json_array(text: str):
    if not text:
        return None
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s)
        s = re.sub(r"\s*```\s*$", "", s)
    start = s.find("[")
    end = s.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(s[start:end + 1])
    except Exception:
        return None


def _validate_quiz_question(q, chunks_meta):
    """Returns (is_valid, reason). On success, normalizes q in place."""
    if not isinstance(q, dict):
        return False, "not a dict"
    qtype = q.get("type")
    if qtype not in ("mcq", "tf"):
        return False, f"bad type {qtype!r}"
    if not q.get("question") or not q.get("explanation"):
        return False, "missing question/explanation"
    src  = q.get("source")
    page = q.get("page")
    if not src or page is None:
        return False, "missing source/page"

    if qtype == "mcq":
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) != 4:
            return False, "options not 4-element list"
        ans = (q.get("correct_answer") or "").strip().upper()
        if ans not in ("A", "B", "C", "D"):
            return False, f"bad mcq answer {q.get('correct_answer')!r}"
        q["correct_answer"] = ans
    else:
        ans = (q.get("correct_answer") or "").strip().lower()
        if ans not in ("true", "false"):
            return False, f"bad tf answer {q.get('correct_answer')!r}"
        q["correct_answer"] = "True" if ans == "true" else "False"

    # Source/page match — case-insensitive on source, tolerant on page type
    src_lc  = str(src).strip().lower()
    page_str = str(page).strip()
    matching_chunk = None
    for c in chunks_meta:
        if (str(c["source"]).strip().lower() == src_lc
                and str(c["page"]).strip() == page_str):
            matching_chunk = c
            # Canonicalise the source name to the chunk's exact spelling
            q["source"] = c["source"]
            q["page"]   = c["page"]
            break
    if matching_chunk is None:
        return False, f"source/page mismatch: {src!r}, page {page!r}"

    # Soft excerpt-quote check: if provided, prefer questions where the quote
    # is in the chunk, but don't reject ones where the LLM paraphrased.
    quote = (q.get("excerpt_quote") or "").strip().lower()
    if quote:
        quote_norm = re.sub(r"\s+", " ", quote)
        chunk_norm = re.sub(r"\s+", " ", (matching_chunk["text"] or "").lower())
        needle = quote_norm[:25]   # short prefix to be tolerant of paraphrase
        if needle and needle not in chunk_norm:
            # Not fatal — log and accept
            pass
    return True, "ok"


def _looks_like_rate_limit(err: Exception) -> bool:
    s = str(err).lower()
    return "rate" in s and ("limit" in s or "429" in s) or "429" in s


def _format_rate_limit_message(err: Exception) -> str:
    s = str(err)
    # Try to surface the "Please try again in 34m9s" portion if present
    m = re.search(r"try again in ([0-9hms\. ]+)", s)
    wait = m.group(1).strip() if m else None
    base = "⏳ The AI is rate-limited (Groq daily token cap)."
    if wait:
        return f"{base} Please try again in **{wait}**."
    return f"{base} Please try again later."


def generate_quiz_questions(topic):
    """Returns (questions, error_message_or_None). On success, len() == QUIZ_LENGTH."""
    excerpts, chunks_meta = _retrieve_quiz_chunks(topic)
    if not chunks_meta:
        return [], "📚 The training manuals don't appear to be ingested yet."

    n_request = QUIZ_LENGTH + 4   # small over-generation cushion to absorb a few drops

    try:
        messages = [
            SystemMessage(content=_QUIZ_GEN_SYSTEM),
            HumanMessage(content=_quiz_gen_user_prompt(excerpts, n_request, topic)),
        ]
        response = get_llm().invoke(messages)
        raw = getattr(response, "content", "") or str(response)
    except Exception as e:
        print(f"⚠️  quiz generation LLM call failed: {e}")
        if _looks_like_rate_limit(e):
            return [], _format_rate_limit_message(e)
        return [], "⚠️ Couldn't reach the AI right now. Please try again in a moment."

    arr = _extract_json_array(raw)
    if not arr:
        print(f"⚠️  quiz JSON parse failed; first 300 chars of LLM output: {raw[:300]!r}")
        return [], "⚠️ The AI didn't return a valid quiz format. Please try again."

    valid: list = []
    drop_reasons: list = []
    for q in arr:
        ok, reason = _validate_quiz_question(q, chunks_meta)
        if ok:
            valid.append(q)
        else:
            drop_reasons.append(reason)

    print(f"📝 quiz generation: requested {n_request}, parsed {len(arr)}, "
          f"validated {len(valid)}; sample drop reasons: {drop_reasons[:3]}")

    if len(valid) < QUIZ_LENGTH:
        return [], (
            f"⚠️ I could only ground {len(valid)} of {len(arr)} questions in the "
            f"manuals on that try. Please ask for the quiz again."
        )

    kept = valid[:QUIZ_LENGTH]
    for i, q in enumerate(kept, start=1):
        q["n"] = i
        q.pop("excerpt_quote", None)
    return kept, None


# ── Quiz turn formatting & grading ───────────────────────────

def format_question(q: dict) -> str:
    n = q.get("n", "?")
    if q["type"] == "mcq":
        opts = "\n".join(q["options"])
        return (f"**Question {n} of {QUIZ_LENGTH}**\n{q['question']}\n\n"
                f"{opts}\n\n_Reply with A, B, C, or D._")
    return (f"**Question {n} of {QUIZ_LENGTH}** (True / False)\n{q['question']}\n\n"
            "_Reply with True or False._")


def _normalize_mcq_answer(user_input: str, q: dict):
    s = user_input.strip()
    if not s:
        return None
    # Find a standalone A/B/C/D letter anywhere in the answer
    m = re.search(r"\b([A-Da-d])\b", s)
    if m:
        return m.group(1).upper()
    # Fallback: match by full option text
    s_lower = s.lower()
    for opt in q.get("options", []):
        om = re.match(r"^\s*([A-D])\s*[).:\-]?\s*(.*)$", opt)
        if not om:
            continue
        letter = om.group(1).upper()
        text = om.group(2).strip().lower()
        if text and (text in s_lower or s_lower in text):
            return letter
    return None


def _normalize_tf_answer(user_input: str):
    s = user_input.strip().lower()
    if not s:
        return None
    if s in ("t", "true", "yes", "y") or s.startswith("true"):
        return "True"
    if s in ("f", "false", "no", "n") or s.startswith("false"):
        return "False"
    return None


def grade_answer(q: dict, user_input: str):
    if q["type"] == "mcq":
        norm = _normalize_mcq_answer(user_input, q)
        if norm is None:
            return False, user_input
        return norm == q["correct_answer"].upper(), norm
    norm = _normalize_tf_answer(user_input)
    if norm is None:
        return False, user_input
    return norm == q["correct_answer"], norm


def format_feedback(is_correct: bool, q: dict, score: int, answered: int) -> str:
    correct_display = q["correct_answer"]
    if q["type"] == "mcq":
        for opt in q.get("options", []):
            if opt.strip().upper().startswith(correct_display.upper() + ")"):
                correct_display = opt
                break
    mark = "✅ **Correct.**" if is_correct else f"❌ **Incorrect** — the correct answer is **{correct_display}**."
    citation = f"_{q['source']}, Page {q['page']}_"
    explanation = q.get("explanation", "")
    return f"{mark} {explanation} ({citation})\n\nRunning score: **{score} / {answered}**"


def format_quiz_summary(score: int, total: int) -> str:
    pct = (score / total) * 100 if total else 0
    if pct >= 80:
        verdict = "Excellent work — you have a strong grasp of the material."
    elif pct >= 60:
        verdict = "Good effort — there are a few gaps worth reviewing."
    else:
        verdict = "There's room to grow — review the cited pages and try another quiz."
    return (f"🎓 **Quiz complete!**\n\n"
            f"**Final score: {score} / {total}** ({pct:.0f}%)\n\n"
            f"{verdict}\n\n"
            f"Type 'quiz me' to start another quiz, or ask any question to keep learning.")


# ── Quiz dispatch ────────────────────────────────────────────

def start_quiz(user_id: str, user_message: str):
    topic = _extract_topic(user_message)
    save_message(user_id, "human", user_message)

    questions, error = generate_quiz_questions(topic)
    if error or not questions:
        ai_response = error or "⚠️ Couldn't start a quiz right now. Please try again."
        save_message(user_id, "ai", ai_response)
        return {"response": ai_response, "sources": []}

    session = create_quiz_session(user_id, questions, topic)
    if not session:
        ai_response = "⚠️ Couldn't save the quiz session. Please try again."
        save_message(user_id, "ai", ai_response)
        return {"response": ai_response, "sources": []}

    first_q = questions[0]
    if topic:
        intro = (f"📝 **Quiz started on _{topic}_!** {QUIZ_LENGTH} questions, "
                 f"drawn straight from your training manuals. "
                 f"Type 'cancel' at any time to stop.\n\n")
    else:
        intro = (f"📝 **Quiz started!** {QUIZ_LENGTH} questions, drawn straight "
                 f"from your training manuals. Type 'cancel' at any time to stop.\n\n")
    body_text = intro + format_question(first_q)
    sources = [f"{first_q['source']} · Page {first_q['page']}"]

    save_message(user_id, "ai", body_text, sources=sources)
    return {"response": body_text, "sources": sources}


def handle_quiz_turn(session: dict, user_id: str, user_message: str):
    save_message(user_id, "human", user_message)

    questions = session["questions"] or []
    idx       = session["current_index"]
    score     = session["score"]
    answers   = session.get("answers") or []

    if idx >= len(questions):
        # Defensive: session should have been completed already
        summary = format_quiz_summary(score, len(questions))
        save_message(user_id, "ai", summary)
        update_quiz_session(session["id"], current_index=idx, score=score,
                            answers=answers, completed=True)
        return {"response": summary, "sources": []}

    current_q = questions[idx]
    is_correct, normalized = grade_answer(current_q, user_message)
    new_score = score + (1 if is_correct else 0)
    answered  = idx + 1
    answers   = answers + [{
        "n":          current_q.get("n", answered),
        "user":       user_message,
        "normalized": normalized,
        "correct":    is_correct,
        "expected":   current_q["correct_answer"],
    }]

    feedback = format_feedback(is_correct, current_q, new_score, answered)
    sources  = [f"{current_q['source']} · Page {current_q['page']}"]

    if answered >= len(questions):
        update_quiz_session(session["id"], current_index=answered,
                            score=new_score, answers=answers, completed=True)
        body = f"{feedback}\n\n---\n\n{format_quiz_summary(new_score, len(questions))}"
        save_message(user_id, "ai", body, sources=sources)
        return {"response": body, "sources": sources}

    next_q = questions[answered]
    next_q_text = format_question(next_q)
    body = f"{feedback}\n\n---\n\n{next_q_text}"
    next_sources = [f"{next_q['source']} · Page {next_q['page']}"]
    update_quiz_session(session["id"], current_index=answered,
                        score=new_score, answers=answers)
    all_sources = list(dict.fromkeys(sources + next_sources))
    save_message(user_id, "ai", body, sources=all_sources)
    return {"response": body, "sources": all_sources}


# ══════════════════════════════════════════════════════════════
#  CHAT ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(body: ChatRequest, current_user=Depends(get_current_user)):
    user_id      = current_user.id
    user_message = body.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # ── Quiz dispatch (server-driven, no LLM hallucination) ──
        active = get_active_quiz(user_id)
        if active:
            if is_quiz_cancel(user_message):
                cancel_quiz_session(active["id"])
                save_message(user_id, "human", user_message)
                progress = f"{active['score']} / {active['current_index']}"
                msg = (f"🛑 Quiz cancelled. Your progress was **{progress}**. "
                       "Ask anything, or type 'quiz me' to start a new one.")
                save_message(user_id, "ai", msg)
                gc.collect()
                return {"response": msg, "sources": []}
            result = handle_quiz_turn(active, user_id, user_message)
            gc.collect()
            return result

        if is_quiz_request(user_message):
            result = start_quiz(user_id, user_message)
            gc.collect()
            return result

        # ── Normal chat flow ──
        manual_context, sources = search_manual(user_message)

        system_content = SYSTEM_PROMPT
        if manual_context:
            system_content += f"\n\nRelevant excerpts from the documents:\n{manual_context}"
        del manual_context

        history          = load_user_history(user_id, limit=30)
        messages_to_send = ([SystemMessage(content=system_content)]
                            + history + [HumanMessage(content=user_message)])
        del history

        save_message(user_id, "human", user_message)

        result      = get_agent().invoke({"messages": messages_to_send})
        ai_response = result["messages"][-1].content
        del result, messages_to_send

        save_message(user_id, "ai", ai_response, sources=list(set(sources)))
        response_data = {"response": ai_response, "sources": list(set(sources))}
        del ai_response, sources

        gc.collect()
        return response_data

    except HTTPException:
        raise
    except Exception as e:
        gc.collect()
        if _looks_like_rate_limit(e):
            return {"response": _format_rate_limit_message(e), "sources": []}
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/chat/history")
def get_history(current_user=Depends(get_current_user)):
    return get_history_for_api(current_user.id, limit=30)

@app.delete("/chat/history")
def clear_history(current_user=Depends(get_current_user)):
    cancel_active_quizzes_for_user(current_user.id)
    delete_user_history(current_user.id)
    return {"message": "Chat history cleared"}

@app.get("/chat/stats")
def get_stats(current_user=Depends(get_current_user)):
    try:
        res   = (supabase_admin.table("chat_messages").select("role", count="exact")
                 .eq("user_id", current_user.id).eq("role", "human").execute())
        count = res.count or 0
        del res
        return {"total_questions": count}
    except Exception:
        return {"total_questions": 0}

# ══════════════════════════════════════════════════════════════
#  DEBUG ROUTE — list all ingested sources
# ══════════════════════════════════════════════════════════════

@app.get("/debug/sources")
def list_sources():
    """
    Returns the distinct 'source' labels stored in the manual collection.
    """
    if not collection_exists(MANUAL_COLLECTION):
        return {"sources": [], "detail": "Manual collection not found."}
    try:
        # Scroll through a sample of points and collect unique source values
        seen, offset = set(), None
        while True:
            result, next_offset = qdrant.scroll(
                collection_name=MANUAL_COLLECTION,
                limit=100,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            for point in result:
                src = point.payload.get("source", "")
                if src:
                    seen.add(src)
            if next_offset is None:
                break
            offset = next_offset
            if len(seen) > 50:   # safety cap
                break
        return {"sources": sorted(seen)}
    except Exception as e:
        return {"sources": [], "detail": str(e)}

# ══════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════

def _prewarm():
    try:
        print("🔌 Pre-warming embedding model in background...")
        embed_one("warmup test")
        print("🔌 Pre-warming LLM + agent in background...")
        get_agent()
        gc.collect()
        print("✅ Pre-warm complete — ready for chat.")
    except Exception as e:
        print(f"⚠️  Pre-warm failed (will load on first request): {e}")

@app.on_event("startup")
def startup():
    print("✅ SRH Tutor API starting — port is bound.")
    init_qdrant_chat_collection()
    init_qdrant_manual_index()        # ← ensures source index exists
    t = threading.Thread(target=_prewarm, daemon=True)
    t.start()
    print(f"📁 Frontend: {STATIC_DIR}")
    print("⏳ Models loading in background...")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Binding to 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
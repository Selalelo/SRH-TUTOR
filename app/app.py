from dotenv import load_dotenv
import os
load_dotenv()

os.environ["HF_TOKEN"]                = os.getenv("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"]  = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"]  = "error"

import warnings, logging, gc, time, uuid, threading
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
        _llm = ChatGroq(model="llama-3.3-70b-versatile", timeout=30, max_retries=1)
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

SYSTEM_PROMPT = """You are a professional, compassionate tutor for SPLA training modules.
You have access to two training documents:
  • SPLA031: Sexual & Reproductive Health (SRH) Training Manual (2024)
  • Chronic Disease & Lifestyle (CDL) training document

Answer questions ONLY from the provided training documents.

CITATION RULES — read carefully:
- Each retrieved excerpt is shown to you as `[<Source>, Page <N>]: <text>`. The source name in the bracket is the EXACT document the excerpt came from.
- When you cite, your citation MUST name a source that actually appears in the excerpts I gave you this turn. Do NOT cite a manual whose name is not in the excerpts.
- Quote the source name verbatim as it appears in the bracket (e.g. "the SPLA031 manual" or "the CDL notes 2022").
- If the excerpts don't cover the question, say so plainly — do not invent a citation.

Teaching style:
- Ground explanations in the provided document excerpts
- Use clear, clinical, respectful language
- Explain concepts step by step
- Be non-judgmental, inclusive, evidence-based, and practical

You have 4 modes:
1. EXPLAIN  → explain using document excerpts + your own clear examples
2. QUIZ     → run a structured 20-question quiz (see QUIZ MODE rules below)
3. EXERCISE → case-study or scenario-based task
4. REVIEW   → give constructive feedback on the trainee's written answer

QUIZ MODE rules (use these whenever the student asks for a quiz, test, or to be quizzed):
- The quiz has exactly 20 questions, asked ONE AT A TIME.
- Mix question styles: multiple-choice (with options A–D), short-answer, and true/false.
- Cover BOTH manuals across the 20 questions — aim for a roughly even split between SRH (SPLA031) and CDL topics. Do NOT ask all 20 questions from a single manual.
- If the student asked for a quiz on a SPECIFIC topic, scope all 20 questions to that topic and use whichever manual(s) cover it.
- When you reveal each answer, cite the manual the question is actually drawn from — and that manual must appear in the excerpts for that turn.
- Format every question as:
    **Question N of 20**
    <the question, with options A–D on separate lines if multiple-choice>
- After the student answers, respond with:
    ✅ Correct  /  ❌ Incorrect — <brief explanation with citation>
    Running score: X / N answered so far
  Then immediately ask the next question.
- After Question 20 is answered, give a final score (e.g. "Final score: 17 / 20"), a 1–2 sentence summary of strong vs. weak areas, and offer to explain any answers in detail.
- Never reveal more than one question at a time. Never list all 20 at once.

After explaining a topic in EXPLAIN mode, ask if they want a quiz or exercise to test understanding.
Keep non-quiz responses concise.
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
#  CHAT ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(body: ChatRequest, current_user=Depends(get_current_user)):
    user_id      = current_user.id
    user_message = body.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # ── Search across both books with source-balanced retrieval ──
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
        raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

@app.get("/chat/history")
def get_history(current_user=Depends(get_current_user)):
    return get_history_for_api(current_user.id, limit=30)

@app.delete("/chat/history")
def clear_history(current_user=Depends(get_current_user)):
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
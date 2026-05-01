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

# ── Module → source label mapping ────────────────────────────
# These must exactly match the labels entered during ingestion.
# Update CDL_SOURCE to match what you typed when ingesting the CDL book.
SRH_SOURCE = os.getenv("SRH_SOURCE", "SPLA031 Sexual & Reproductive Health Training Manual 2024")
CDL_SOURCE = os.getenv("CDL_SOURCE", "")   # set this in your .env or Render env vars

MODULE_SOURCES = {
    "srh": SRH_SOURCE,
    "cdl": CDL_SOURCE,
}

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
#  MANUAL SEARCH  (now module-aware)
# ══════════════════════════════════════════════════════════════

def search_manual(query: str, n_results: int = 3, source_filter: str = None):
    """
    Search the manual collection.
    If source_filter is provided, only chunks whose 'source' payload matches
    that value are returned — preventing cross-book contamination.
    """
    if not collection_exists(MANUAL_COLLECTION):
        raise HTTPException(status_code=503,
            detail="📚 Manual not ingested yet. Run: python ingest_srh.py")
    try:
        query_vector = embed_one(query)

        # Build optional source filter
        query_filter = None
        if source_filter:
            query_filter = Filter(must=[
                FieldCondition(key="source", match=MatchValue(value=source_filter))
            ])

        results = qdrant.query_points(
            collection_name=MANUAL_COLLECTION,
            query=query_vector,
            limit=n_results,
            with_payload=True,
            query_filter=query_filter      # ← only chunks from the right book
        )
        if not results.points:
            return "", []
        parts, sources = [], []
        for hit in results.points:
            parts.append(f"[Page {hit.payload['page']}]:\n{hit.payload['text']}")
            sources.append(f"Page {hit.payload['page']}")
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

def load_user_history(user_id: str, limit: int = 8):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content").eq("user_id", user_id)
               .order("created_at", desc=False).limit(limit).execute())
        messages = []
        for row in res.data:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                messages.append(AIMessage(content=row["content"]))
        del res
        return messages
    except Exception as e:
        print(f"⚠️  History load failed: {e}")
        return []

def get_history_for_api(user_id: str, limit: int = 30):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content, sources, created_at")
               .eq("user_id", user_id)
               .order("created_at", desc=False).limit(limit).execute())
        data = res.data or []
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
#  SYSTEM PROMPTS  (one per module)
# ══════════════════════════════════════════════════════════════

SRH_SYSTEM_PROMPT = """You are a professional, compassionate tutor for SPLA training modules.
You have access to the SPLA031: Sexual & Reproductive Health (SRH) Training Manual (2024).

Answer questions ONLY from the SRH training document.
Always cite the document and page number you are referencing.
If asked about something not covered in the SRH document, politely say it is outside
the scope of the available SRH training materials.

Teaching style:
- Ground explanations in the provided document excerpts
- Use clear, clinical, respectful language
- Explain concepts step by step
- Be non-judgmental and inclusive
- Always cite e.g. "As the SPLA031 manual states on page 34..."

You have 4 modes:
1. EXPLAIN  → explain using document excerpts + your own clear examples
2. QUIZ     → give 1 exam-style question, wait for answer before revealing it
3. EXERCISE → case-study or scenario-based task
4. REVIEW   → give constructive feedback on the trainee's written answer

After explaining, ask if they want a quiz or exercise to test understanding.
Keep responses concise.
"""

CDL_SYSTEM_PROMPT = """You are a professional, knowledgeable tutor for SPLA training modules.
You have access to the Chronic Disease & Lifestyle (CDL) training document.

Answer questions ONLY from the CDL training document.
Always cite the document and page number you are referencing.
If asked about something not covered in the CDL document, politely say it is outside
the scope of the available CDL training materials.

Teaching style:
- Ground explanations in the provided document excerpts
- Use clear, clinical, respectful language
- Explain concepts step by step
- Be evidence-based and practical
- Always cite e.g. "As the CDL manual states on page 12..."

You have 4 modes:
1. EXPLAIN  → explain using document excerpts + your own clear examples
2. QUIZ     → give 1 exam-style question, wait for answer before revealing it
3. EXERCISE → case-study or scenario-based task
4. REVIEW   → give constructive feedback on the trainee's written answer

After explaining, ask if they want a quiz or exercise to test understanding.
Keep responses concise.
"""

MODULE_PROMPTS = {
    "srh": SRH_SYSTEM_PROMPT,
    "cdl": CDL_SYSTEM_PROMPT,
}

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
    module: str = "srh"   # "srh" or "cdl" — sent by the frontend

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
    module       = body.module if body.module in MODULE_SOURCES else "srh"

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    try:
        # ── Source filter: only search chunks from the active module's book ──
        source_filter = MODULE_SOURCES.get(module) or None
        manual_context, sources = search_manual(user_message, n_results=3,
                                                source_filter=source_filter)

        # ── System prompt for this module ──
        system_content = MODULE_PROMPTS.get(module, SRH_SYSTEM_PROMPT)
        if manual_context:
            system_content += f"\n\nRelevant excerpts from the document:\n{manual_context}"
        del manual_context

        history          = load_user_history(user_id, limit=8)
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
    Use this to confirm the exact string to put in MODULE_SOURCES / .env.
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
    if not CDL_SOURCE:
        print("⚠️  CDL_SOURCE env var is not set — CDL module will search all books.")
        print("   Run GET /debug/sources to find the correct label, then set CDL_SOURCE.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Binding to 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
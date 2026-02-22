from dotenv import load_dotenv
import os
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN", "")
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import time, uuid
from typing import List, Union, Optional
from typing_extensions import TypedDict
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Paths
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

# ══════════════════════════════════════════════════════════════
#  CONNECTIONS  (lightweight — no model loading here)
# ══════════════════════════════════════════════════════════════

_url         = os.getenv("SUPABASE_URL")
_anon_key    = os.getenv("SUPABASE_ANON_KEY")
_service_key = os.getenv("SUPABASE_SERVICE_KEY", _anon_key)

supabase: Client       = create_client(_url, _anon_key)
supabase_admin: Client = create_client(_url, _service_key)

qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

MANUAL_COLLECTION = "srh_manual"
CHAT_COLLECTION   = "srh_chat"
VECTOR_SIZE       = 384

# ── Embedder: truly lazy — loaded on first use ────────────────
_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        print("🔌 Loading embedding model (first use)...")
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
        print("✅ Embedding model ready.")
    return _embedder

# ── LLM: lazy too ────────────────────────────────────────────
_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(model="llama-3.3-70b-versatile")
    return _llm

# ── LangGraph agent: lazy ─────────────────────────────────────
_agent = None

def get_agent():
    global _agent
    if _agent is None:
        class AgentState(TypedDict):
            messages: List[Union[HumanMessage, SystemMessage, AIMessage]]

        def process(state: AgentState) -> AgentState:
            response = get_llm().invoke(state["messages"])
            state["messages"].append(AIMessage(content=response.content))
            return state

        graph = StateGraph(AgentState)
        graph.add_node("process", process)
        graph.add_edge(START, "process")
        graph.add_edge("process", END)
        _agent = graph.compile()
    return _agent

# ══════════════════════════════════════════════════════════════
#  FASTAPI APP  — created immediately so port can bind
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="SRH Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(str(STATIC_DIR / "index.html"))

@app.get("/health")
def health():
    """Render health check — responds instantly, no heavy deps needed."""
    return {"status": "ok"}

# ══════════════════════════════════════════════════════════════
#  QDRANT HELPERS
# ══════════════════════════════════════════════════════════════

def init_qdrant_chat_collection():
    try:
        existing = [c.name for c in qdrant.get_collections().collections]
        if CHAT_COLLECTION not in existing:
            qdrant.create_collection(
                collection_name=CHAT_COLLECTION,
                vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
            )
            qdrant.create_payload_index(
                collection_name=CHAT_COLLECTION,
                field_name="timestamp",
                field_schema=PayloadSchemaType.FLOAT
            )
            print(f"✅ Qdrant collection '{CHAT_COLLECTION}' created.")
    except Exception as e:
        print(f"⚠️  Qdrant init warning: {e}")

# ══════════════════════════════════════════════════════════════
#  AUTH HELPER
# ══════════════════════════════════════════════════════════════

def get_current_user(authorization: str = Header(...)):
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    token = authorization.split(" ")[1]
    try:
        user = supabase.auth.get_user(token)
        if not user or not user.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user.user
    except Exception:
        raise HTTPException(status_code=401, detail="Could not validate token")

# ══════════════════════════════════════════════════════════════
#  MANUAL SEARCH
# ══════════════════════════════════════════════════════════════

def search_manual(query: str, n_results: int = 4):
    existing = [c.name for c in qdrant.get_collections().collections]
    if MANUAL_COLLECTION not in existing:
        raise HTTPException(
            status_code=503,
            detail="📚 Manual not ingested yet. Run: python ingest_srh.py"
        )
    try:
        query_vector = get_embedder().encode(query).tolist()
        results = qdrant.query_points(
            collection_name=MANUAL_COLLECTION,
            query=query_vector,
            limit=n_results,
            with_payload=True
        )
        if not results.points:
            return "", []
        context = ""
        sources = []
        for hit in results.points:
            page = hit.payload["page"]
            text = hit.payload["text"]
            context += f"\n[Page {page}]:\n{text}\n"
            sources.append(f"Page {page}")
        return context, sources
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
            "id": msg_id, "user_id": user_id,
            "role": role, "content": content,
            "sources": sources if sources else []
        }).execute()
    except Exception as e:
        print(f"⚠️  Supabase insert failed: {e}")
    try:
        vector = get_embedder().encode(content).tolist()
        qdrant.upsert(
            collection_name=CHAT_COLLECTION,
            points=[PointStruct(
                id=msg_id, vector=vector,
                payload={"user_id": user_id, "role": role,
                         "content": content, "timestamp": time.time()}
            )]
        )
    except Exception as e:
        print(f"⚠️  Qdrant upsert failed: {e}")

def load_user_history(user_id: str, limit: int = 20):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content")
               .eq("user_id", user_id)
               .order("created_at", desc=False)
               .limit(limit).execute())
        messages = []
        for row in res.data:
            if row["role"] == "human":
                messages.append(HumanMessage(content=row["content"]))
            elif row["role"] == "ai":
                messages.append(AIMessage(content=row["content"]))
        return messages
    except Exception as e:
        print(f"⚠️  History load failed: {e}")
        return []

def get_history_for_api(user_id: str, limit: int = 30):
    try:
        res = (supabase_admin.table("chat_messages")
               .select("role, content, sources, created_at")
               .eq("user_id", user_id)
               .order("created_at", desc=False)
               .limit(limit).execute())
        return res.data or []
    except Exception as e:
        print(f"⚠️  get_history_for_api failed: {e}")
        return []

def delete_user_history(user_id: str):
    try:
        supabase_admin.table("chat_messages").delete().eq("user_id", user_id).execute()
    except Exception as e:
        print(f"⚠️  Supabase delete failed: {e}")
    try:
        qdrant.delete(
            collection_name=CHAT_COLLECTION,
            points_selector=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
        )
    except Exception as e:
        print(f"⚠️  Qdrant delete failed: {e}")

def update_last_seen(user_id: str):
    try:
        supabase_admin.table("profiles").update(
            {"last_seen": "now()"}
        ).eq("id", user_id).execute()
    except Exception:
        pass

# ══════════════════════════════════════════════════════════════
#  SYSTEM PROMPT
# ══════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a professional, compassionate tutor for the SPLA031 Sexual &
Reproductive Health (SRH) Training Manual (2024), focused on physiology and health education.

You ONLY answer questions from the SRH Training Manual topics:
- Male and female reproductive anatomy & physiology
- Menstrual cycle, ovulation, fertilisation, conception
- Contraception methods and family planning
- STIs and HIV/AIDS
- Pregnancy, antenatal and postnatal care
- Sexual health, consent, healthy relationships
- Puberty and adolescent development
- Reproductive rights and gender-based health equity

If asked anything unrelated, politely redirect the trainee back to SRH topics.

Teaching style:
- Ground explanations in provided manual excerpts
- Use clear, clinical, respectful language
- Explain physiology step by step
- Be non-judgmental and inclusive
- Cite page numbers when relevant e.g. "As the manual explains on page 34..."

You have 4 modes:
1. EXPLAIN  → explain using manual excerpts + your own clear examples
2. QUIZ     → give 1 exam-style question, wait for answer before revealing it
3. EXERCISE → case-study or scenario-based task
4. REVIEW   → give constructive feedback on the trainee's written answer

After explaining, ask if they want a quiz or exercise to test understanding.
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
        res = supabase.auth.sign_up({
            "email": body.email, "password": body.password,
            "options": {"data": {"full_name": body.full_name}}
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")
        return {"message": "Account created! Please check your email to confirm.", "user_id": res.user.id}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/auth/signin")
def signin(body: SignInRequest):
    try:
        res = supabase.auth.sign_in_with_password({"email": body.email, "password": body.password})
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        update_last_seen(res.user.id)
        return {
            "access_token": res.session.access_token,
            "user": {"id": res.user.id, "email": res.user.email,
                     "full_name": res.user.user_metadata.get("full_name", "")}
        }
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
        supabase.auth.resend({"type": "signup", "email": body.email})
        return {"message": "Confirmation email resent! Check your inbox."}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not resend: {str(e)}")

@app.post("/auth/signout")
def signout(current_user=Depends(get_current_user)):
    supabase.auth.sign_out()
    return {"message": "Signed out successfully"}

@app.get("/auth/me")
def get_me(current_user=Depends(get_current_user)):
    try:
        profile = (supabase_admin.table("profiles")
                   .select("full_name, email, role, created_at, last_seen")
                   .eq("id", current_user.id).single().execute())
        return {
            "id": current_user.id, "email": current_user.email,
            "full_name": profile.data.get("full_name", ""),
            "role": profile.data.get("role", "trainee"),
            "created_at": profile.data.get("created_at"),
            "last_seen": profile.data.get("last_seen"),
        }
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

    manual_context, sources = search_manual(user_message, n_results=4)

    system_content = SYSTEM_PROMPT
    if manual_context:
        system_content += f"\n\nRelevant manual excerpts:\n{manual_context}"

    history = load_user_history(user_id, limit=10)
    messages_to_send = (
        [SystemMessage(content=system_content)]
        + history
        + [HumanMessage(content=user_message)]
    )

    save_message(user_id, "human", user_message)
    result      = get_agent().invoke({"messages": messages_to_send})
    ai_response = result["messages"][-1].content
    save_message(user_id, "ai", ai_response, sources=list(set(sources)))

    return {"response": ai_response, "sources": list(set(sources))}

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
        res = (supabase_admin.table("chat_messages")
               .select("role", count="exact")
               .eq("user_id", current_user.id)
               .eq("role", "human").execute())
        return {"total_questions": res.count or 0}
    except Exception:
        return {"total_questions": 0}

# ══════════════════════════════════════════════════════════════
#  STARTUP — minimal, just init qdrant collection
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    print("✅ SRH Tutor API starting — port is bound.")
    init_qdrant_chat_collection()
    print(f"📁 Frontend: {STATIC_DIR}")
    print("⏳ Embedding model will load on first chat request.")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    print(f"🚀 Binding to 0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
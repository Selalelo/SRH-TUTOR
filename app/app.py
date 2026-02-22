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
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════
#  CONNECTIONS
# ══════════════════════════════════════════════════════════════

# Supabase — two clients:
# 1. service client  → bypasses RLS, used for all DB writes (backend only)
# 2. anon client     → used only for auth (sign in / sign up / verify token)
_url      = os.getenv("SUPABASE_URL")
_anon_key = os.getenv("SUPABASE_ANON_KEY")
# Service role key bypasses RLS — get it from Supabase → Settings → API → service_role
_service_key = os.getenv("SUPABASE_SERVICE_KEY", _anon_key)  # falls back to anon if not set

supabase: Client       = create_client(_url, _anon_key)     # auth operations
supabase_admin: Client = create_client(_url, _service_key)  # DB read/write (bypasses RLS)

# Qdrant — vector search on the manual + semantic chat context
qdrant = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

MANUAL_COLLECTION = "srh_manual"
CHAT_COLLECTION   = "srh_chat"
VECTOR_SIZE       = 384

def init_qdrant_chat_collection():
    """Create Qdrant chat collection for semantic search (if not exists)."""
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

# ══════════════════════════════════════════════════════════════
#  AUTH HELPER
# ══════════════════════════════════════════════════════════════

def get_current_user(authorization: str = Header(...)):
    """Validate Supabase JWT from Authorization: Bearer <token>."""
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
#  MANUAL SEARCH (Qdrant RAG)
# ══════════════════════════════════════════════════════════════

def search_manual(query: str, n_results: int = 4):
    """Search the ingested SRH manual PDF for relevant excerpts."""
    # Check collection exists before querying
    existing = [c.name for c in qdrant.get_collections().collections]
    if MANUAL_COLLECTION not in existing:
        raise HTTPException(
            status_code=503,
            detail=(
                "📚 The SRH manual has not been ingested yet. "
                "Please run: python ingest_srh.py — and point it at your PDF file. "
                "Then restart the server."
            )
        )
    try:
        query_vector = embedder.encode(query).tolist()
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
#  CHAT HISTORY  ← Supabase is the source of truth
# ══════════════════════════════════════════════════════════════

def save_message(user_id: str, role: str, content: str, sources: list = []):
    """
    Save a chat message to:
      1. Supabase  → permanent record, queryable, shown in dashboard
      2. Qdrant    → vector index for semantic retrieval
    """
    msg_id = str(uuid.uuid4())

    # 1️⃣  Supabase (primary store)
    try:
        supabase_admin.table("chat_messages").insert({
            "id":       msg_id,
            "user_id":  user_id,
            "role":     role,
            "content":  content,
            "sources":  sources if sources else []
        }).execute()
    except Exception as e:
        print(f"⚠️  Supabase insert failed: {e}")

    # 2️⃣  Qdrant (semantic vector store)
    try:
        vector = embedder.encode(content).tolist()
        qdrant.upsert(
            collection_name=CHAT_COLLECTION,
            points=[PointStruct(
                id=msg_id,
                vector=vector,
                payload={
                    "user_id":   user_id,
                    "role":      role,
                    "content":   content,
                    "timestamp": time.time()
                }
            )]
        )
    except Exception as e:
        print(f"⚠️  Qdrant upsert failed: {e}")


def load_user_history(user_id: str, limit: int = 20):
    """
    Load recent chat history from Supabase (ordered, reliable).
    Returns LangChain message objects for the LLM context window.
    """
    try:
        res = (
            supabase_admin.table("chat_messages")
            .select("role, content")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
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
    """
    Return raw message dicts for the /chat/history endpoint.
    Used by the frontend to render conversation on load.
    """
    try:
        res = (
            supabase_admin.table("chat_messages")
            .select("role, content, sources, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .limit(limit)
            .execute()
        )
        return res.data or []
    except Exception as e:
        print(f"⚠️  get_history_for_api failed: {e}")
        return []


def delete_user_history(user_id: str):
    """
    Delete all messages for a user from:
      1. Supabase  (hard delete)
      2. Qdrant    (filter delete)
    """
    # Supabase
    try:
        supabase_admin.table("chat_messages").delete().eq("user_id", user_id).execute()
    except Exception as e:
        print(f"⚠️  Supabase delete failed: {e}")

    # Qdrant
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
#  LANGGRAPH AGENT
# ══════════════════════════════════════════════════════════════

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, SystemMessage, AIMessage]]

llm = ChatGroq(model="llama-3.3-70b-versatile")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response.content))
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
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

# ── Pydantic Models ───────────────────────────────────────────
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
    """
    Register a new trainee.
    Supabase automatically creates a row in auth.users.
    Our trigger (from supabase_schema.sql) then creates a row in public.profiles.
    """
    try:
        res = supabase.auth.sign_up({
            "email":   body.email,
            "password": body.password,
            "options": {"data": {"full_name": body.full_name}}
        })
        if not res.user:
            raise HTTPException(status_code=400, detail="Signup failed")
        return {
            "message": "Account created! Please check your email to confirm your address.",
            "user_id": res.user.id
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/auth/signin")
def signin(body: SignInRequest):
    """Sign in and receive a JWT access token."""
    try:
        res = supabase.auth.sign_in_with_password({
            "email":    body.email,
            "password": body.password
        })
        if not res.user:
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Update last_seen in profiles table
        update_last_seen(res.user.id)

        return {
            "access_token": res.session.access_token,
            "user": {
                "id":        res.user.id,
                "email":     res.user.email,
                "full_name": res.user.user_metadata.get("full_name", "")
            }
        }
    except Exception as e:
        error_msg = str(e)
        # Give a helpful message for the most common cause
        if "Email not confirmed" in error_msg or "email_not_confirmed" in error_msg:
            detail = "Please confirm your email address first. Check your inbox for a confirmation link — or disable email confirmation in your Supabase dashboard (Authentication → Settings)."
        elif "Invalid login credentials" in error_msg:
            detail = "Incorrect email or password. Please try again."
        else:
            detail = f"Sign in failed: {error_msg}"
        raise HTTPException(status_code=401, detail=detail)


@app.post("/auth/resend-confirmation")
def resend_confirmation(body: SignInRequest):
    """
    Resend the confirmation email for an unconfirmed account.
    Only requires email — password is validated to prevent abuse.
    """
    try:
        res = supabase.auth.resend({
            "type":  "signup",
            "email": body.email,
        })
        return {"message": "Confirmation email resent! Check your inbox and click the link within 24 hours."}
    except Exception as e:
        error_msg = str(e)
        if "User already registered" in error_msg:
            return {"message": "This email is already confirmed. You can sign in now."}
        raise HTTPException(status_code=400, detail=f"Could not resend email: {error_msg}")


@app.post("/auth/signout")
def signout(current_user=Depends(get_current_user)):
    supabase.auth.sign_out()
    return {"message": "Signed out successfully"}


@app.get("/auth/me")
def get_me(current_user=Depends(get_current_user)):
    """Get the logged-in user's profile from Supabase."""
    try:
        profile = (
            supabase_admin.table("profiles")
            .select("full_name, email, role, created_at, last_seen")
            .eq("id", current_user.id)
            .single()
            .execute()
        )
        return {
            "id":         current_user.id,
            "email":      current_user.email,
            "full_name":  profile.data.get("full_name", ""),
            "role":       profile.data.get("role", "trainee"),
            "created_at": profile.data.get("created_at"),
            "last_seen":  profile.data.get("last_seen"),
        }
    except Exception:
        return {
            "id":        current_user.id,
            "email":     current_user.email,
            "full_name": current_user.user_metadata.get("full_name", ""),
            "role":      "trainee"
        }

# ══════════════════════════════════════════════════════════════
#  CHAT ROUTES
# ══════════════════════════════════════════════════════════════

@app.post("/chat")
def chat(body: ChatRequest, current_user=Depends(get_current_user)):
    """
    Main chat endpoint:
    1. Search the SRH manual (Qdrant RAG)
    2. Load user's recent history (Supabase)
    3. Call the AI agent (Groq)
    4. Save both messages to Supabase + Qdrant
    """
    user_id      = current_user.id
    user_message = body.message.strip()

    if not user_message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # 1. RAG — search manual
    manual_context, sources = search_manual(user_message, n_results=4)

    # 2. Build system prompt with manual context
    system_content = SYSTEM_PROMPT
    if manual_context:
        system_content += f"\n\nRelevant manual excerpts:\n{manual_context}"

    # 3. Load history from Supabase
    history = load_user_history(user_id, limit=10)

    messages_to_send = (
        [SystemMessage(content=system_content)]
        + history
        + [HumanMessage(content=user_message)]
    )

    # 4. Save user message to Supabase + Qdrant
    save_message(user_id, "human", user_message)

    # 5. Run the AI agent
    result       = agent.invoke({"messages": messages_to_send})
    ai_response  = result["messages"][-1].content

    # 6. Save AI response to Supabase + Qdrant
    save_message(user_id, "ai", ai_response, sources=list(set(sources)))

    return {
        "response": ai_response,
        "sources":  list(set(sources))
    }


@app.get("/chat/history")
def get_history(current_user=Depends(get_current_user)):
    """
    Return the user's last 30 messages from Supabase.
    Called by the frontend on load to restore the conversation.
    """
    return get_history_for_api(current_user.id, limit=30)


@app.delete("/chat/history")
def clear_history(current_user=Depends(get_current_user)):
    """Delete all chat messages for the current user (Supabase + Qdrant)."""
    delete_user_history(current_user.id)
    return {"message": "Chat history cleared"}


@app.get("/chat/stats")
def get_stats(current_user=Depends(get_current_user)):
    """Return basic stats for the user's dashboard."""
    try:
        res = (
            supabase_admin.table("chat_messages")
            .select("role", count="exact")
            .eq("user_id", current_user.id)
            .eq("role", "human")
            .execute()
        )
        total_questions = res.count or 0
        return {"total_questions": total_questions}
    except Exception:
        return {"total_questions": 0}

# ══════════════════════════════════════════════════════════════
#  STARTUP
# ══════════════════════════════════════════════════════════════

@app.on_event("startup")
def startup():
    init_qdrant_chat_collection()
    print("✅ SRH Tutor API is running.")
    print(f"📁 Serving frontend from: {STATIC_DIR}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import uuid

# ── Suppress noisy logs ───────────────────────────────────────
import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from srh_embedder import embed  # Pure ONNX, no Rust needed

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
COLLECTION_NAME = "srh_manual"
VECTOR_SIZE     = 384     # all-MiniLM-L6-v2 ONNX output size
CHUNK_SIZE      = 500     # characters per chunk
CHUNK_OVERLAP   = 100     # overlap between chunks
BATCH_SIZE      = 50      # chunks per Qdrant upload batch

# ══════════════════════════════════════════════════════════════
#  CONNECTIONS
# ══════════════════════════════════════════════════════════════

print("\n🔌 Connecting to Qdrant...")
try:
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    qdrant.get_collections()
    print("✅ Qdrant connected.")
except Exception as e:
    print(f"❌ Could not connect to Qdrant: {e}")
    print("   Check QDRANT_URL and QDRANT_API_KEY in your .env file.")
    sys.exit(1)

print("🔌 Loading ONNX embedding model...")
try:
    test = embed(["test"])
    print(f"✅ Embedding model ready. Vector size: {len(test[0])}")
    del test
except Exception as e:
    print(f"❌ Could not load embedding model: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  STEP 1 — Setup Qdrant Collection
# ══════════════════════════════════════════════════════════════

def init_collection():
    """
    Returns: 'add'    — collection exists, append new doc to it
             'fresh'  — collection deleted and recreated
             False    — user cancelled
    """
    existing = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in existing:
        count = qdrant.count(collection_name=COLLECTION_NAME).count
        print(f"\n📦 Collection '{COLLECTION_NAME}' already exists ({count} vectors).")
        print("   Options:")
        print("   [1] ADD  — append this document (keep existing data)")
        print("   [2] REPLACE — delete everything and start fresh")
        print("   [3] CANCEL")
        answer = input("   Choose 1, 2, or 3: ").strip()
        if answer == "1":
            print("➕ Adding new document to existing collection...")
            return "add"
        elif answer == "2":
            qdrant.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted old collection.")
        else:
            print("⏭️  Cancelled.")
            return False
    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created.")
    return "fresh"

# ══════════════════════════════════════════════════════════════
#  STEP 2 — Extract Text from PDF
# ══════════════════════════════════════════════════════════════

def extract_pdf(path: str):
    print(f"\n📄 Opening PDF: {path}")
    try:
        doc = fitz.open(path)
    except Exception as e:
        print(f"❌ Could not open PDF: {e}")
        sys.exit(1)
    pages   = []
    skipped = 0
    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({"page": page_num + 1, "text": text})
        else:
            skipped += 1
    doc.close()
    print(f"✅ Extracted {len(pages)} pages with text.")
    if skipped > 0:
        print(f"   ℹ️  Skipped {skipped} blank/image-only pages.")
    if len(pages) == 0:
        print("❌ No text found in PDF. It may be scanned images — OCR needed.")
        sys.exit(1)
    return pages

# ══════════════════════════════════════════════════════════════
#  STEP 3 — Split into Overlapping Chunks
# ══════════════════════════════════════════════════════════════

def chunk_pages(pages):
    print(f"\n✂️  Chunking (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = []
    for page_data in pages:
        text     = page_data["text"]
        page_num = page_data["page"]
        start    = 0
        while start < len(text):
            chunk = text[start : start + CHUNK_SIZE].strip()
            if chunk:
                chunks.append({
                    "text":     chunk,
                    "page":     page_num,
                    "chunk_id": len(chunks)
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    print(f"✅ Created {len(chunks)} chunks from {len(pages)} pages.")
    return chunks

# ══════════════════════════════════════════════════════════════
#  STEP 4 — Embed and Upload to Qdrant
# ══════════════════════════════════════════════════════════════

def upload_chunks(chunks, source_label: str = "SPLA031 Sexual & Reproductive Health Training Manual 2024"):
    total    = len(chunks)
    uploaded = 0
    failed   = 0
    print(f"\n🚀 Uploading {total} chunks to Qdrant (batches of {BATCH_SIZE})...")

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Embed with ONNX
        try:
            vectors = embed(texts)
        except Exception as e:
            print(f"\n   ⚠️  Embedding failed for batch {i}–{i+len(batch)}: {e}")
            failed += len(batch)
            continue

        # Build Qdrant points
        points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vectors[j],
                payload={
                    "text":     batch[j]["text"],
                    "page":     batch[j]["page"],
                    "chunk_id": batch[j]["chunk_id"],
                    "source":   source_label
                }
            )
            for j in range(len(batch))
        ]

        # Upload
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            uploaded += len(batch)
        except Exception as e:
            print(f"\n   ⚠️  Upload failed for batch {i}–{i+len(batch)}: {e}")
            failed += len(batch)
            continue

        # Progress bar
        done   = min(i + BATCH_SIZE, total)
        filled = int(30 * done / total)
        bar    = "█" * filled + "░" * (30 - filled)
        pct    = int(100 * done / total)
        print(f"   [{bar}] {pct}%  ({done}/{total})", end="\r")
        time.sleep(0.05)

    print(f"\n\n✅ Upload complete: {uploaded} uploaded, {failed} failed.")
    return uploaded

# ══════════════════════════════════════════════════════════════
#  STEP 5 — Verify
# ══════════════════════════════════════════════════════════════

def verify():
    try:
        info  = qdrant.get_collection(COLLECTION_NAME)
        count = qdrant.count(collection_name=COLLECTION_NAME).count
        print(f"\n🔍 Verification:")
        print(f"   Collection : {COLLECTION_NAME}")
        print(f"   Vectors    : {count}")
        print(f"   Status     : {info.status}")
        if count == 0:
            print("   ⚠️  No vectors found — something went wrong.")
        else:
            print(f"   ✅ Manual is searchable in Qdrant.")
    except Exception as e:
        print(f"   ⚠️  Could not verify: {e}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  📚 SPLA Document Ingestion Tool")
    print(f"  Model : all-MiniLM-L6-v2 (ONNX)")
    print("=" * 55)

    # Get PDF path from argument or prompt
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1].strip().strip("'\"")
    else:
        pdf_path = input("\nEnter the full path to your PDF file:\n> ").strip().strip("'\"")

    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print("⚠️  Warning: file does not end in .pdf — continuing anyway...")

    mode = init_collection()
    if mode:
        # Ask for a friendly document label (used in source citations)
        default_label = os.path.splitext(os.path.basename(pdf_path))[0]
        label = input(f"\nDocument label for citations (default: {default_label}):\n> ").strip()
        if not label:
            label = default_label

        pages    = extract_pdf(pdf_path)
        chunks   = chunk_pages(pages)
        uploaded = upload_chunks(chunks, source_label=label)
        if uploaded > 0:
            verify()
            action = "Added to" if mode == "add" else "Created"
            print(f"\n🎉 {action} collection with {uploaded} chunks from '{label}'!")
            print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload\n")
        else:
            print("\n❌ No chunks uploaded. Check errors above.")
            sys.exit(1)
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
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import fitz  # PyMuPDF
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
COLLECTION_NAME = "srh_manual"
VECTOR_SIZE     = 384
CHUNK_SIZE      = 500    # characters per chunk
CHUNK_OVERLAP   = 100    # overlap between chunks to preserve context
BATCH_SIZE      = 50     # how many chunks to upload at once

# ══════════════════════════════════════════════════════════════
#  CONNECTIONS
# ══════════════════════════════════════════════════════════════
print("\n🔌 Connecting to Qdrant...")
try:
    qdrant = QdrantClient(
        url=os.getenv("QDRANT_URL"),
        api_key=os.getenv("QDRANT_API_KEY")
    )
    # Test connection
    qdrant.get_collections()
    print("✅ Qdrant connected.")
except Exception as e:
    print(f"❌ Could not connect to Qdrant: {e}")
    print("   Check QDRANT_URL and QDRANT_API_KEY in your .env file.")
    sys.exit(1)

print("🔌 Loading embedding model (all-MiniLM-L6-v2)...")
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    print("✅ Embedding model loaded.")
except Exception as e:
    print(f"❌ Could not load embedding model: {e}")
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
#  STEP 1 — Setup Qdrant Collection
# ══════════════════════════════════════════════════════════════
def init_collection():
    existing = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME in existing:
        print(f"\n⚠️  Collection '{COLLECTION_NAME}' already exists in Qdrant.")
        answer = input("   Re-ingest? This will DELETE existing data (yes/no): ").strip().lower()
        if answer == "yes":
            qdrant.delete_collection(COLLECTION_NAME)
            print(f"🗑️  Deleted old collection '{COLLECTION_NAME}'.")
        else:
            print("⏭️  Skipping ingestion — existing data kept.")
            return False

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE)
    )
    print(f"✅ Collection '{COLLECTION_NAME}' created.")
    return True

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

    pages = []
    skipped = 0

    for page_num, page in enumerate(doc):
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page": page_num + 1,
                "text": text
            })
        else:
            skipped += 1  # blank or image-only page

    doc.close()

    print(f"✅ Extracted {len(pages)} pages with text.")
    if skipped > 0:
        print(f"   ℹ️  Skipped {skipped} blank/image-only pages.")
    if len(pages) == 0:
        print("❌ No text found in PDF.")
        print("   This PDF may be scanned images. You would need an OCR tool first.")
        sys.exit(1)

    return pages

# ══════════════════════════════════════════════════════════════
#  STEP 3 — Split Pages into Overlapping Chunks
# ══════════════════════════════════════════════════════════════
def chunk_text(pages):
    print(f"\n✂️  Chunking text (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    chunks = []

    for page_data in pages:
        text     = page_data["text"]
        page_num = page_data["page"]
        start    = 0

        while start < len(text):
            end   = start + CHUNK_SIZE
            chunk = text[start:end].strip()

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
def upload_chunks(chunks):
    total   = len(chunks)
    print(f"\n🚀 Uploading {total} chunks to Qdrant in batches of {BATCH_SIZE}...")

    uploaded = 0
    failed   = 0

    for i in range(0, total, BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        # Embed the batch
        try:
            vectors = embedder.encode(
                texts,
                show_progress_bar=False,
                convert_to_numpy=True
            ).tolist()
        except Exception as e:
            print(f"   ⚠️  Embedding failed for batch {i}–{i+BATCH_SIZE}: {e}")
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
                    "source":   "SPLA031 Sexual & Reproductive Health Training Manual 2024"
                }
            )
            for j in range(len(batch))
        ]

        # Upload to Qdrant
        try:
            qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
            uploaded += len(batch)
        except Exception as e:
            print(f"   ⚠️  Upload failed for batch {i}–{i+BATCH_SIZE}: {e}")
            failed += len(batch)
            continue

        # Progress
        done_so_far = min(i + BATCH_SIZE, total)
        bar_len     = 30
        filled      = int(bar_len * done_so_far / total)
        bar         = "█" * filled + "░" * (bar_len - filled)
        pct         = int(100 * done_so_far / total)
        print(f"   [{bar}] {pct}%  ({done_so_far}/{total} chunks)", end="\r")

        time.sleep(0.1)  # be kind to the API rate limits

    print(f"\n\n✅ Upload complete: {uploaded} chunks uploaded, {failed} failed.")
    return uploaded

# ══════════════════════════════════════════════════════════════
#  STEP 5 — Verify the Collection
# ══════════════════════════════════════════════════════════════
def verify_collection():
    try:
        info  = qdrant.get_collection(COLLECTION_NAME)
        count = qdrant.count(collection_name=COLLECTION_NAME).count
        print(f"\n🔍 Verification:")
        print(f"   Collection : {COLLECTION_NAME}")
        print(f"   Vectors    : {count}")
        print(f"   Status     : {info.status}")

        if count == 0:
            print("   ⚠️  No vectors found — something went wrong with the upload.")
        else:
            print(f"   ✅ Ready! Your SRH manual is searchable in Qdrant.")
    except Exception as e:
        print(f"   ⚠️  Could not verify collection: {e}")

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 55)
    print("  📚 SRH Manual Ingestion — SPLA031")
    print("=" * 55)

    # Get PDF path
    if len(sys.argv) > 1:
        # Allow passing path as command-line argument
        pdf_path = sys.argv[1].strip()
    else:
        pdf_path = input("\nEnter the full path to your PDF file:\n> ").strip()

    # Remove accidental quotes (common when dragging file into terminal)
    pdf_path = pdf_path.strip("'\"")

    if not os.path.exists(pdf_path):
        print(f"\n❌ File not found: {pdf_path}")
        print("   Make sure the path is correct and try again.")
        sys.exit(1)

    if not pdf_path.lower().endswith(".pdf"):
        print("⚠️  Warning: file does not have a .pdf extension. Continuing anyway...")

    # Run pipeline
    if init_collection():
        pages    = extract_pdf(pdf_path)
        chunks   = chunk_text(pages)
        uploaded = upload_chunks(chunks)

        if uploaded > 0:
            verify_collection()
            print("\n🎉 Ingestion complete! You can now start the tutor:")
            print("   uvicorn app:app --host 0.0.0.0 --port 8000 --reload\n")
        else:
            print("\n❌ Ingestion failed — no chunks were uploaded.")
            sys.exit(1)
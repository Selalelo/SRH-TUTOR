"""
build.py — runs during Render build step BEFORE the server starts.
Downloads and caches the fastembed model (~50MB, much lighter than sentence-transformers).
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

print("📦 Pre-downloading embedding model during build...")
try:
    from fastembed import TextEmbedding
    model  = TextEmbedding("BAAI/bge-small-en-v1.5")
    result = list(model.embed(["warmup test"]))
    print(f"✅ fastembed model ready. Vector size: {len(result[0])}")
    del model, result
except Exception as e:
    print(f"⚠️  Model download failed: {e}")
    print("   Will attempt download at runtime.")
"""
build.py — runs during Render's build step BEFORE the server starts.
Downloads and caches the embedding model so it's ready at runtime.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

print("📦 Pre-downloading embedding model during build...")
try:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    # Run a test encode to confirm it works
    test = model.encode("test", convert_to_numpy=True)
    print(f"✅ Model downloaded and cached. Vector size: {len(test)}")
    del model, test
except Exception as e:
    print(f"⚠️  Model download failed: {e}")
    print("   Model will attempt to download at runtime instead.")
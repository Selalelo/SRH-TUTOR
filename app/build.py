"""
build.py — runs during Render build step BEFORE the server starts.
Pre-downloads the ONNX embedding model so it's cached at runtime.
"""
import os
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

print("📦 Pre-downloading ONNX embedding model during build...")
try:
    from srh_embedder import embed_one
    result = embed_one("warmup test")
    print(f"✅ ONNX model ready. Vector size: {len(result)}")
    del result
except Exception as e:
    print(f"⚠️  Model download failed: {e}")
    print("   Will attempt at runtime.")
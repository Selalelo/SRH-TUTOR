"""
embedder.py — lightweight ONNX-based embedder, no Rust, no PyTorch.
Downloads all-MiniLM-L6-v2 ONNX model from HuggingFace on first use.
"""
import os
import numpy as np
from pathlib import Path
from huggingface_hub import hf_hub_download
import onnxruntime as ort
from tokenizers import Tokenizer

CACHE_DIR  = Path(os.getenv("HF_HOME", Path.home() / ".cache" / "srh_embedder"))
MODEL_REPO = "sentence-transformers/all-MiniLM-L6-v2"
ONNX_FILE  = "onnx/model.onnx"
TOK_FILE   = "tokenizer.json"
MAX_LEN    = 128

_session   = None
_tokenizer = None

def _load():
    global _session, _tokenizer
    if _session is not None:
        return
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("🔌 Downloading ONNX embedding model...")
    model_path = hf_hub_download(repo_id=MODEL_REPO, filename=ONNX_FILE, cache_dir=CACHE_DIR)
    tok_path   = hf_hub_download(repo_id=MODEL_REPO, filename=TOK_FILE,  cache_dir=CACHE_DIR)
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    _session   = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    _tokenizer = Tokenizer.from_file(tok_path)
    _tokenizer.enable_truncation(max_length=MAX_LEN)
    _tokenizer.enable_padding(length=MAX_LEN)
    print("✅ ONNX embedding model ready.")

def _mean_pool(token_embeddings, attention_mask):
    mask = attention_mask[:, :, np.newaxis].astype(np.float32)
    summed = np.sum(token_embeddings * mask, axis=1)
    counts = np.clip(mask.sum(axis=1), a_min=1e-9, a_max=None)
    return summed / counts

def _normalise(v):
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.clip(norm, a_min=1e-9, a_max=None)

def embed(texts: list[str]) -> list[list[float]]:
    """Embed a list of strings. Returns list of 384-dim vectors."""
    _load()
    encoded = _tokenizer.encode_batch(texts)
    input_ids      = np.array([e.ids              for e in encoded], dtype=np.int64)
    attention_mask = np.array([e.attention_mask   for e in encoded], dtype=np.int64)
    token_type_ids = np.array([e.type_ids         for e in encoded], dtype=np.int64)
    outputs = _session.run(None, {
        "input_ids":      input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
    })
    pooled     = _mean_pool(outputs[0], attention_mask)
    normalised = _normalise(pooled)
    return normalised.tolist()

def embed_one(text: str) -> list[float]:
    """Embed a single string."""
    return embed([text])[0]
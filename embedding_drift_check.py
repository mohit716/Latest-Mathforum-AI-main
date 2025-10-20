#!/usr/bin/env python3

"""
Extended Embedding Drift Check for Corpus RAG vs Generic Corpora
==============================================================
- Load all batch_* ChromaDB directories from a vectorstore ensemble
- Compute self-similarity, cross-similarity, and math neighborhood cohesion
- Compare embeddings from a fine-tuned LLM using mean pooling
- Save fine-tuned embeddings to .npz
- Produce overlay histograms for global and math-heavy subsets
- Output embedding shift matrix as CSV

- A Generic Chroma vectordb can be supplied, or a generic model (either a directory or a huggingface model to be downloaded).  If the finetuned model is a PEFT, its underlying base model must be supplied via --finetuned-base-model.

Run with:
python embedding_drift_check.py \
  --corpus-chroma vectorstore \
  --collection-prefix vectordb \
  --generic-chroma generic_store \
  --generic-model "intfloat/e5-base-v2" \
  --finetuned-model llama3-finetuned \
  --finetuned-base-model "NousResearch/Nous-Hermes-2-Mistral-7B-DPO" \
  --embed-limit 2000 \
  --plots \
  --outdir drift_report \
  --cache-dir embedding_cache \
  --align-mode project --aligned-dim 512
"""

import os
import re
import math
import json
import csv
import random
import argparse
import traceback
from tqdm import tqdm, trange
from pathlib import Path
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, ProcessPoolExecutor
import hashlib
import tempfile
import numpy as np
import chromadb
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModel
import torch
from transformers import AutoConfig
from huggingface_hub import model_info, HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError
from peft import PeftModel  
from itertools import repeat

# ----------------------- Math pattern detection ---------------------------

MATH_PATTERNS = [
    r"\$[^$]+\$",               # inline LaTeX
    r"\\\[[\s\S]*?\\\]",        # display math
    r"\\begin\{equation\}[\s\S]*?\\end\{equation\}",
    r"\\frac\{", r"\\sqrt\{", r"\\sum", r"\\int", r"\\lim",
    r"[∀∃∑∫∞≈≃≅≡≤≥√±÷×⋅∠°′″→⇒⇔≈≠∈∉∩∪⊂⊆⊇⊃∅∴∵∝]",
    r"[^\w]\^[^\s]",            # exponent markers
    r"_[^\s\w]",                # subscripts
]
MATH_REGEX = re.compile("|".join(MATH_PATTERNS))

def mathiness_score(text: str) -> float:
    if not text:
        return 0.0
    matches = MATH_REGEX.findall(text)
    L = max(1.0, math.log(len(text) + 1, 10))
    return len(matches) / L

# ------------------------- Cache -----------------------------
   
def _slugify(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]

def _collection_cache_paths(cache_dir: str, collection_name: str, max_docs: int):
    base = os.path.join(cache_dir, f"{_slugify(collection_name)}__{collection_name}__{max_docs}")
    return base + ".npz", base + ".jsonl", base + ".meta.json"

def _save_collection_cache(cache_dir: str, name: str, max_docs: int, ids, embeddings: np.ndarray, documents):
    """
    Atomically save per-collection caches:
    - embeddings -> .npz
    - (id, doc) pairs -> .jsonl
    - small metadata -> .meta.json

    We stage to temp files in the target directory and then os.replace(...) to guarantee atomicity.
    """
    try:
        npz_path, jsonl_path, meta_path = _collection_cache_paths(cache_dir, name, max_docs)
        outdir = os.path.dirname(npz_path) or "."
        os.makedirs(outdir, exist_ok=True)

        # ---- embeddings (.npz) atomic write ----
        with tempfile.NamedTemporaryFile(dir=outdir, suffix=".npz", delete=False) as tf:
            tmp_npz = tf.name
        try:
            # Write directly to the temp file path, then replace
            np.savez_compressed(tmp_npz, embeddings=np.asarray(embeddings, dtype=np.float32))
            os.replace(tmp_npz, npz_path)
        except Exception as e:
            safe_print(f"cache_save:{name}:npz", e)
            # Best-effort cleanup
            try:
                if os.path.exists(tmp_npz):
                    os.remove(tmp_npz)
            except Exception as e2:
                safe_print(f"cache_save:{name}:npz_cleanup", e2)
            raise

        # ---- jsonl (ids + documents) atomic write ----
        with tempfile.NamedTemporaryFile(dir=outdir, suffix=".jsonl", delete=False, mode="w", encoding="utf-8") as tf:
            tmp_jsonl = tf.name
            for i, d in zip(ids, documents):
                tf.write(json.dumps({"id": i, "doc": d if d is not None else ""}))
                tf.write("\n")
            tf.flush()
            os.fsync(tf.fileno())
        try:
            os.replace(tmp_jsonl, jsonl_path)
        except Exception as e:
            safe_print(f"cache_save:{name}:jsonl", e)
            try:
                if os.path.exists(tmp_jsonl):
                    os.remove(tmp_jsonl)
            except Exception as e2:
                safe_print(f"cache_save:{name}:jsonl_cleanup", e2)
            raise

        # ---- meta.json atomic write ----
        meta_payload = {"name": name, "max_docs": max_docs, "count": len(ids)}
        with tempfile.NamedTemporaryFile(dir=outdir, suffix=".meta.json", delete=False, mode="w", encoding="utf-8") as tf:
            tmp_meta = tf.name
            tf.write(json.dumps(meta_payload))
            tf.flush()
            os.fsync(tf.fileno())
        try:
            os.replace(tmp_meta, meta_path)
        except Exception as e:
            safe_print(f"cache_save:{name}:meta", e)
            try:
                if os.path.exists(tmp_meta):
                    os.remove(tmp_meta)
            except Exception as e2:
                safe_print(f"cache_save:{name}:meta_cleanup", e2)
            raise

        print(f"[cache saved:{name}]")
    except Exception as e:
        safe_print(f"cache_save:{name}", e)
        raise

def _try_load_collection_cache(cache_dir: str, name: str, max_docs: int):
    npz_path, jsonl_path, meta_path = _collection_cache_paths(cache_dir, name, max_docs)
    if not (os.path.exists(npz_path) and os.path.exists(jsonl_path) and os.path.exists(meta_path)):
        return None
    try:
        embs = np.load(npz_path, allow_pickle=False)["embeddings"]
        ids, docs = [], []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids.append(rec["id"])
                docs.append(rec.get("doc", ""))
        return {"ids": ids[:max_docs], "embeddings": embs[:max_docs], "documents": docs[:max_docs]}
    except Exception as e:
        safe_print("load_cache(collection)", e)
        return None
        
def _savefig_atomic(fig, final_path: str):
    """
    Save a matplotlib figure atomically:
    - Write to a temp file in the same directory
    - Replace the final path in a single operation
    Ensures we never leave a partial PNG if interrupted.
    """
    try:
        outdir = os.path.dirname(final_path) or "."
        os.makedirs(outdir, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=outdir, suffix=".png", delete=False) as tf:
            tmp_path = tf.name
        fig.savefig(tmp_path)  # write the full file
        fig.canvas.draw()      # strengthen flush in some backends
        os.replace(tmp_path, final_path)
        print(f"[written] {final_path}")
    except Exception as e:
        safe_print(f"savefig_atomic:{final_path}", e)
        # Best-effort cleanup of the tmp, in case savefig succeeded but replace failed
        try:
            if 'tmp_path' in locals() and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception as e2:
            safe_print("savefig_atomic:cleanup", e2)
        raise        
        
def _skip_if_exists(path: str) -> bool:
    try:
        return os.path.exists(path) and os.path.getsize(path) > 0
    except Exception:
        return False
        
# ----------------------- Utilities ---------------------------

def safe_print(prefix: str, e: Exception):
    print(f"[{prefix}] {e}")
    traceback.print_exc()

def cosine_sim_from_distance(d: np.ndarray) -> np.ndarray:
    return 1.0 - d

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def _is_mathy(entry: str, thresh: float) -> bool:
    try:
        return mathiness_score(entry) >= thresh
    except Exception:
        return False

def mathy_indices(texts: List[str], thresh: float, workers: int = -1) -> List[int]:
    """
    Return indices of texts whose mathiness_score >= thresh.

    Uses a ProcessPoolExecutor with only top-level callables to avoid pickling issues.
    """
    workers = (os.cpu_count() or 1) if workers == -1 else max(1, workers)

    # NOTE: _is_mathy must be top-level for ProcessPool pickling to work.
    with ProcessPoolExecutor(max_workers=workers) as ex:
        flags_iter = ex.map(_is_mathy, texts, repeat(thresh))
        flags = list(tqdm(flags_iter, total=len(texts),
                          desc="Scanning for math-heavy docs", unit="doc"))

    return [i for i, f in enumerate(flags) if f]

def align_same_dim(
    Xa: np.ndarray,
    Xb: np.ndarray,
    mode: str = "truncate",          # "truncate" | "pad"
    dim: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (A,B) with the same feature dimension.
    """
    Da, Db = Xa.shape[1], Xb.shape[1]
    if Da == Db:
        return Xa, Xb

    if mode == "truncate":
        d = dim if dim is not None else min(Da, Db)
        if d <= 0:
            raise ValueError("dim must be positive for truncate mode.")
        A = Xa[:, :d] if Da >= d else Xa
        B = Xb[:, :d] if Db >= d else Xb
        return A, B

    if mode == "pad":
        d = dim if dim is not None else max(Da, Db)
        if d <= 0:
            raise ValueError("dim must be positive for pad mode.")
        A = Xa if Da == d else np.pad(Xa, ((0,0),(0, d - Da)))
        B = Xb if Db == d else np.pad(Xb, ((0,0),(0, d - Db)))
        return A, B

    tqdm.write(f"[align_same_dim] Unknown mode: {mode}")
    return Xa, Xb

def paired_cosine_stats(
    Xa: np.ndarray,
    Xb: np.ndarray,
    samples: int,
    align_mode: str = "truncate",   # "truncate" | "pad" | "project"
    aligned_dim: Optional[int] = None,
    proj_seed: int = 0,
) -> np.ndarray:
    """
    Compute cosine similarity between *paired* rows from two different spaces.

    Parameters
    ----------
    Xa, Xb : np.ndarray
        Shapes (N, Da), (N, Db). May differ in feature size.
        Assumes rows i correspond to the same item (paired).
    samples : int
        Number of row indices to sample uniformly without replacement.
    align_mode : str
        How to reconcile Da != Db:
          - 'truncate': use first d=min(Da,Db) (or --aligned-dim if provided).
          - 'pad': zero-pad the shorter to match the longer.
          - 'project': random Gaussian projection of both to a common d.
    aligned_dim : Optional[int]
        Target dimension for 'truncate' or 'project'. If None:
          - 'truncate': d=min(Da,Db)
          - 'project': d=min(min(Da,Db), 1024)
    proj_seed : int
        RNG seed for 'project' mode to ensure reproducibility.

    Returns
    -------
    np.ndarray
        1D array of cosine similarities of length 'samples' (or N if samples>=N).
    """
    n = min(len(Xa), len(Xb))
    if n == 0:
        return np.array([], dtype=np.float32)

    idx = sample_indices(n, samples)
    A = Xa[idx].astype(np.float32, copy=False)
    B = Xb[idx].astype(np.float32, copy=False)

    Da = A.shape[1]
    Db = B.shape[1]

    def _l2_normalize(X: np.ndarray) -> np.ndarray:
        denom = np.linalg.norm(X, axis=1, keepdims=True)
        denom = np.where(denom == 0, 1e-12, denom)
        return X / denom

    if Da == Db:
        A = _l2_normalize(A)
        B = _l2_normalize(B)
        return np.sum(A * B, axis=1)

    # Align feature sizes
    if align_mode == "truncate":
        d = aligned_dim if aligned_dim is not None else min(Da, Db)
        if d <= 0:
            raise ValueError("aligned_dim must be positive for truncate mode.")
        A = A[:, :d] if Da >= d else A
        B = B[:, :d] if Db >= d else B

    elif align_mode == "pad":
        d = max(Da, Db)
        if Da < d:
            A = np.pad(A, ((0, 0), (0, d - Da)))
        if Db < d:
            B = np.pad(B, ((0, 0), (0, d - Db)))

    elif align_mode == "project":
        # Project both to a shared low-ish d to reduce noise
        d_default = min(min(Da, Db), 1024)
        d = aligned_dim if aligned_dim is not None else d_default
        if d <= 0:
            raise ValueError("aligned_dim must be positive for project mode.")
        rng = np.random.default_rng(proj_seed)
        # Gaussian random projection with variance 1/d
        Wa = rng.normal(loc=0.0, scale=1.0 / math.sqrt(d), size=(Da, d)).astype(np.float32)
        Wb = rng.normal(loc=0.0, scale=1.0 / math.sqrt(d), size=(Db, d)).astype(np.float32)
        A = A @ Wa
        B = B @ Wb

    else:
        raise ValueError(f"Unknown align_mode: {align_mode}")

    A = _l2_normalize(A)
    B = _l2_normalize(B)
    sims = np.sum(A * B, axis=1)
    return sims
    
def preflight_model_check(model_id_or_path: str, token: Optional[str] = None, finetuned_base_model: Optional[str] = None):
    """
    Verifies that a model can be accessed before loading weights.

    Local directories:
      - If it contains config.json => treat as a full Transformers model (OK).
      - If it contains adapter_config.json => treat as a PEFT adapter; require finetuned_base_model.
      - Rejects obvious non-Transformers formats (e.g., *.gguf).
    HF Hub:
      - Confirms repo exists and that config is loadable (may require token).
    """
    p = Path(model_id_or_path)
    if p.exists() and p.is_dir():
        names = {x.name for x in p.iterdir()} if p.exists() else set()

        # Disallow gguf here (transformers AutoModel can’t load it)
        if any(n.lower().endswith(".gguf") for n in names):
            raise RuntimeError(
                f"Local model folder looks like GGUF (found .gguf). "
                "This script uses transformers AutoModel and cannot load GGUF. "
                "Provide a Transformers-format model directory or a Hub repo id."
            )

        # Full model present?
        if "config.json" in names:
            print(f"[preflight] Local model directory OK (found config.json): {model_id_or_path}")
            return "local"

        # PEFT adapter present?
        if "adapter_config.json" in names or "adapter_model.safetensors" in names:
            if not finetuned_base_model:
                raise RuntimeError(
                    "Local path appears to be a PEFT/LoRA adapter (found adapter_config.json). "
                    "Pass --finetuned-base-model with a compatible base HF model id or local directory, "
                    "or merge the adapter into a full model and point --finetuned-model to the merged dir."
                )
            print("[preflight] Detected PEFT adapter; will load on top of base model.")
            return "local-adapter"

        # Otherwise, missing required metadata
        raise RuntimeError(
            "Local model folder missing required files. Expected either:\n"
            "  • A full Transformers model (config.json present), or\n"
            "  • A PEFT adapter (adapter_config.json present) + --finetuned-base-model.\n"
            f"Found entries: {sorted(names)}"
        )

    # HF Hub check
    try:
        info = model_info(model_id_or_path, token=token)
        print(f"[preflight] Hugging Face model found: {info.modelId}")
        if info.private:
            print("[preflight] Model is private; token authentication may be required.")
        if info.gated:
            print("[preflight] Model is gated; ensure you have accepted its license.")
        try:
            AutoConfig.from_pretrained(model_id_or_path, token=token)
        except TypeError:
            AutoConfig.from_pretrained(model_id_or_path, use_auth_token=token)
        return "hub"
    except RepositoryNotFoundError:
        raise RuntimeError(
            f"Model '{model_id_or_path}' not found on Hugging Face Hub. "
            "Check the name or whether you have access."
        )
    except GatedRepoError:
        raise RuntimeError(
            f"Model '{model_id_or_path}' is gated. "
            "You must accept the license on the model page and provide a valid token."
        )
    except Exception as e:
        raise RuntimeError(
            f"Preflight check failed for model '{model_id_or_path}': {e}"
        )
    
# ----------------------- Chroma loading ---------------------------

def _resolve_model_source(model_id_or_path: str) -> str:
    """
    Returns 'local' if model_id_or_path is a directory that exists,
    otherwise returns the string unchanged (assumed to be a HF repo id).
    """
    if model_id_or_path is None:
        return None
    p = Path(model_id_or_path)
    return "local" if p.exists() and p.is_dir() else "hub"

def load_chroma_embeddings(
    chroma_dir: str,
    collections: Optional[List[str]] = None,
    max_docs: int = 100000,
    cache_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    out = {}
    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        colls = client.list_collections()
        for c in colls:
            name = c.name
            if collections and name not in collections:
                continue
            try:
                # Try cache first
                if cache_dir:
                    cached = _try_load_collection_cache(cache_dir, name, max_docs)
                    if cached is not None:
                        out[name] = cached
                        print(f"[cache hit:{name}] {cached['embeddings'].shape}")
                        continue

                # Fallback: load from Chroma
                col = client.get_collection(name)
                ids, embs, docs = [], [], []
                page = 0
                try:
                    total_items = col.count()
                except AttributeError:
                    total_items = len(col.get(include=[], limit=0)["ids"])
                total_pages = math.ceil(total_items / 5000) if total_items else None

                with tqdm(desc=f"Loading {name}", unit="page", total=total_pages) as pbar:
                    while True:
                        res = col.get(include=["embeddings", "documents"], limit=5000, offset=page * 5000)
                        if not res["ids"]:
                            break
                        ids.extend(res["ids"])
                        if res.get("embeddings") is None:
                            break
                        embs.extend(res["embeddings"])
                        docs.extend(res.get("documents", [""] * len(res["ids"])))
                        page += 1
                        pbar.update(1)
                        if len(ids) >= max_docs:
                            break

                if ids and embs:
                    arr = np.array(embs, dtype=np.float32)[:max_docs]
                    rec = {
                        "ids": ids[:max_docs],
                        "embeddings": arr,
                        "documents": docs[:max_docs]
                    }
                    out[name] = rec
                    print(f"[loaded:{name}] {arr.shape}")
                    # Save to cache atomically
                    if cache_dir:
                        try:
                            _save_collection_cache(cache_dir, name, max_docs, rec["ids"], rec["embeddings"], rec["documents"])
                            print(f"[cache saved:{name}]")
                        except Exception as e:
                            safe_print(f"cache_save:{name}", e)
            except Exception as e:
                safe_print(f"collection:{name}", e)
    except Exception as e:
        safe_print("load_chroma_embeddings", e)
    return out

def load_batches(
    corpus_root: str,
    prefix: str,
    max_docs: int,
    workers: Optional[int] = None,
    cache_dir: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Load all batch_* collections under corpus_root in parallel.

    Parameters
    ----------
    corpus_root : str
        Directory that contains batch_* subdirectories (each a Chroma store).
    prefix : str
        Collection name prefix used when creating the collections, e.g., "vectordb".
    max_docs : int
        Cap per collection.
    workers : Optional[int]
        Thread pool size for loading batches concurrently.
    cache_dir : Optional[str]
        If provided, use per-collection caches (see load_chroma_embeddings).

    Returns
    -------
    Dict[str, Dict[str, Any]]
        Mapping {collection_name: {"ids": [...], "embeddings": np.ndarray, "documents": [...]} }
    """
    batches = sorted(Path(corpus_root).glob("batch_*"))
    combined: Dict[str, Dict[str, Any]] = {}
    workers = workers or max(1, os.cpu_count() or 1)

    def _load_one(bdir: Path) -> Dict[str, Dict[str, Any]]:
        coll_name = f"{prefix}_{bdir.name}"
        return load_chroma_embeddings(
            str(bdir),
            collections=[coll_name],
            max_docs=max_docs,
            cache_dir=cache_dir
        )

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_load_one, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Corpus batches (parallel)", unit="batch"):
            try:
                combined.update(fut.result())
            except Exception as e:
                safe_print(f"corpus_batch:{futures[fut].name}", e)
    return combined

# ----------------------- Similarity computation ---------------------------

def sample_indices(n: int, samples: int) -> np.ndarray:
    if samples >= n:
        return np.arange(n)
    return np.random.choice(n, size=samples, replace=False)

def nearest_neighbor_stats(
    Xa: np.ndarray,
    Xb: np.ndarray,
    k: int,
    samples: int,
    n_jobs: Optional[int] = None,
    align_mode: str = "truncate",        
    aligned_dim: Optional[int] = None,   
    proj_seed: int = 0,                  
) -> np.ndarray:
    """
    kNN within-space when Da==Db; otherwise paired cosine with explicit alignment.

    Returns
    -------
    np.ndarray
        If Da == Db: cosine to the nearest neighbor (k=1) for sampled queries.
        If Da != Db and len(Xa)==len(Xb): paired cosine similarities with alignment.
    """
    Da = Xa.shape[1]
    Db = Xb.shape[1]

    if Da == Db:
        idx = sample_indices(len(Xa), samples)
        nn = NearestNeighbors(
            n_neighbors=min(k, len(Xb)),
            metric="cosine",
            algorithm="brute",
            n_jobs=n_jobs
        )
        nn.fit(Xb)
        dists, _ = nn.kneighbors(Xa[idx], return_distance=True)
        sims = cosine_sim_from_distance(dists)[:, 0]
        return sims
        
    Xa, Xb = align_same_dim(Xa, Xb, mode=align_mode)

    if len(Xa) == len(Xb):
        return paired_cosine_stats(
            Xa, Xb, samples,
            align_mode=align_mode,
            aligned_dim=aligned_dim,
            proj_seed=proj_seed,
        )

    raise ValueError(
        f"Cannot compare embeddings with different feature dimensions (Da={Da}, Db={Db}) "
        f"and different lengths (len(Xa)={len(Xa)}, len(Xb)={len(Xb)}). "
        "Use a learned mapping, CKA, or restrict comparisons to same-space embeddings."
    )

# ----------------------- Plotting ---------------------------

def overlay_histogram(data_dict: Dict[str, np.ndarray], title: str, path: str):
    """
    Draws overlaid histograms and writes the PNG atomically.
    """    
    if _skip_if_exists(path):
        print(f"[skip plot exists] {path}")
        return

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111)
        for label, sims in data_dict.items():
            ax.hist(sims, bins=50, alpha=0.5, label=label, density=True)
        ax.legend()
        ax.set_title(title)
        ax.set_xlabel("Cosine similarity")
        ax.set_ylabel("Density")
        fig.tight_layout()
        _savefig_atomic(fig, path)
    except Exception as e:
        safe_print("overlay_histogram", e)
        raise
    finally:
        plt.close(fig)

# ---------------------------- Embedding --------------------------------

def embed_with_finetuned(
    model_dir: str,
    texts: List[str],
    limit: Optional[int],
    cache_path: str,
    force: bool = False,
    prefer_bf16: bool = False,
    batch_size: int = 8,
    token: Optional[str] = None,
    finetuned_base_model: Optional[str] = None,
) -> np.ndarray:
    """
    Embed texts using a (fine-tuned) HF model via mean pooling.
    Supports three sources:
      (A) Full local model directory (has config.json)
      (B) HF Hub repo id
      (C) Local PEFT adapter dir (adapter_config.json present) + finetuned_base_model
    """
    try:
        if os.path.exists(cache_path) and not force:
            print(f"[cache] Loading cached embeddings from {cache_path}")
            return np.load(cache_path)["embeddings"]

        if limit:
            texts = random.sample(texts, min(limit, len(texts)))

        # Choose dtype
        use_cuda = torch.cuda.is_available()
        if prefer_bf16 and use_cuda and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("[dtype] Using bfloat16")
        else:
            dtype = torch.float16
            print("[dtype] Using float16")

        if use_cuda:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception as e:
                safe_print("tf32 flags", e)

        device = torch.device("cuda:0" if use_cuda else "cpu")
        print(f"[target_device] {device}")

        # Detect adapter vs full model
        model_path = Path(model_dir)
        is_local = model_path.exists() and model_path.is_dir()
        is_adapter = is_local and (
            (model_path / "adapter_config.json").exists() or
            (model_path / "adapter_model.safetensors").exists()
        )
        has_config = is_local and (model_path / "config.json").exists()

        # Load tokenizer + model
        try:
            if is_adapter:
                if not finetuned_base_model:
                    raise RuntimeError(
                        "Adapter directory detected but --finetuned-base-model was not provided."
                    )
                # Tokenizer should come from base (adapter dirs typically lack tokenizer)
                try:
                    tokenizer = AutoTokenizer.from_pretrained(finetuned_base_model, use_fast=True, token=token)
                except TypeError:
                    tokenizer = AutoTokenizer.from_pretrained(finetuned_base_model, use_fast=True, use_auth_token=token)

                # For embeddings we want the base encoder/decoder backbone without LM head
                try:
                    base_backbone = AutoModel.from_pretrained(finetuned_base_model, torch_dtype=dtype, token=token)
                except TypeError:
                    base_backbone = AutoModel.from_pretrained(finetuned_base_model, torch_dtype=dtype, use_auth_token=token)

                # Apply adapter weights
                base_backbone = PeftModel.from_pretrained(base_backbone, model_dir)
                model = base_backbone

            else:
                # Full model or Hub repo
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, token=token)
                except TypeError:
                    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True, use_auth_token=token)

                try:
                    model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype, token=token)
                except TypeError:
                    model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype, use_auth_token=token)

        except Exception as e:
            safe_print("model_load", e)
            raise RuntimeError(
                f"Failed to load model/tokenizer for '{model_dir}'. "
                "If this is a gated/private repo, ensure license acceptance and pass --hf-token. "
                "If this is an adapter dir, also pass --finetuned-base-model."
            ) from e

        model.to(device)
        model.eval()

        try:
            print("[hf_device_map]", getattr(model, "hf_device_map", None))
        except Exception:
            pass

        all_embeddings = []
        try:
            with torch.inference_mode():
                for i in trange(0, len(texts), batch_size, desc="Embedding with model", unit="batch"):
                    batch = texts[i:i+batch_size]
                    enc = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        return_tensors="pt",
                        max_length=512
                    )
                    for k in enc:
                        enc[k] = enc[k].to(device, non_blocking=True)

                    if "encoder_attention_mask" in enc:
                        enc.pop("encoder_attention_mask", None)

                    outputs = model(**enc)
                    pooled = mean_pooling(outputs, enc["attention_mask"])
                    pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                    all_embeddings.append(pooled.detach().cpu().numpy())
        except Exception as e:
            safe_print("embed_with_finetuned:forward", e)
            raise

        all_embeddings = np.vstack(all_embeddings)
        np.savez(cache_path, embeddings=all_embeddings)
        print(f"[cache] Saved embeddings to {cache_path}")
        return all_embeddings

    except Exception as e:
        safe_print("embed_with_finetuned", e)
        raise
    
def embed_with_model(
    model_dir: str,
    texts: List[str],
    limit: Optional[int],
    cache_path: str,
    force: bool = False,
    prefer_bf16: bool = False,
    batch_size: int = 8,
) -> np.ndarray:
    # Thin wrapper around embed_with_finetuned to avoid renaming everywhere
    return embed_with_finetuned(
        model_dir=model_dir,
        texts=texts,
        limit=limit,
        cache_path=cache_path,
        force=force,
        prefer_bf16=prefer_bf16,
        batch_size=batch_size,
    )    

# ----------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-chroma", required=True)
    ap.add_argument("--collection-prefix", default="vectordb")
    ap.add_argument("--generic-chroma", help="Path to a generic Chroma DB. If provided, this takes precedence over --generic-model.")
    ap.add_argument("--generic-model", help="Path or HF id for a base model to embed corpus texts on-the-fly as 'generic'. Used only if --generic-chroma is not provided.")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN"),
                    help="Optional Hugging Face token for gated repos. If omitted, relies on cached auth or public models.")    
    ap.add_argument("--finetuned-model", help="Path or HF id for the fine-tuned LLM to embed corpus texts as 'finetuned'.")
    ap.add_argument("--finetuned-base-model", help="HF id or path of the base model to use when --finetuned-model is a PEFT adapter dir.")
    ap.add_argument("--embed-limit", type=int)
    ap.add_argument("--force-embed", action="store_true")
    ap.add_argument("--samples", type=int, default=500)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-docs", type=int, default=100000)
    ap.add_argument("--math-thresh", type=float, default=0.5)
    ap.add_argument("--plots", action="store_true")
    ap.add_argument("--outdir", default="drift_report")
    ap.add_argument("--align-mode", choices=["truncate", "pad", "project"], default="truncate",
                    help="Alignment strategy when comparing paired embeddings from different spaces.")
    ap.add_argument("--aligned-dim", type=int, default=None,
                    help="Target dimension for 'truncate' or 'project' modes.")
    ap.add_argument("--proj-seed", type=int, default=0,
                    help="Seed for random projection in 'project' mode.")
    ap.add_argument("--jobs", type=int, default=-1,
                    help="Parallelism for CPU-bound tasks (e.g., sklearn NN, loading, mathiness). Use -1 for all cores.")
    ap.add_argument("--cache-dir", default=None,
                    help="Optional directory for per-collection .npz caches to avoid reloading/rehashing.")
    ap.add_argument("--force-cache-rebuild", action="store_true",
                    help="Ignore any existing .npz caches and rebuild from Chroma.")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Disable cache reads if --force-cache-rebuild is set
    effective_cache_dir = None if args.force_cache_rebuild else args.cache_dir
    if effective_cache_dir:
        os.makedirs(effective_cache_dir, exist_ok=True)

    # Where to write new embedding caches
    cache_root = args.cache_dir or args.outdir
    os.makedirs(cache_root, exist_ok=True)

    # ---------- Local helpers for CSV completeness checks ----------
    def _csv_has_required_headers_and_rows(path: str, required_cols: list[str], required_rows: list[str], row_label_col: str = "") -> bool:
        """
        For the square shift matrix CSV:
        - required_cols: exact column names expected (including row label column "")
        - required_rows: row labels that must be present in the first column
        """
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                # Check columns
                cols = r.fieldnames or []
                if cols != required_cols:
                    return False
                # Check all required row labels appear at least once
                have = set()
                for row in r:
                    if row_label_col not in row:
                        return False
                    have.add(row[row_label_col])
                return all(lbl in have for lbl in required_rows)
        except Exception as e:
            safe_print("csv_check:shift_matrix", e)
            return False

    def _pairs_csv_complete(path: str, labels: list[str]) -> bool:
        """
        For the pairs CSV: ensure required columns exist and that every unordered
        pair among labels (including (L,L)) appears at least once in either order.
        """
        required_cols = ["A", "B", "mean_cosine", "relative_drift_pct"]
        try:
            if not os.path.exists(path) or os.path.getsize(path) == 0:
                return False
            with open(path, "r", encoding="utf-8", newline="") as f:
                r = csv.DictReader(f)
                cols = r.fieldnames or []
                if cols != required_cols:
                    return False
                # Build expected unordered pair keys
                expected = set()
                for i, a in enumerate(labels):
                    for j, b in enumerate(labels):
                        if j < i:
                            continue
                        expected.add(tuple(sorted((a, b))))
                seen = set()
                for row in r:
                    a, b = row.get("A"), row.get("B")
                    if a is None or b is None:
                        continue
                    seen.add(tuple(sorted((a, b))))
                # Complete if we've seen all required unordered pairs
                return expected.issubset(seen)
        except Exception as e:
            safe_print("csv_check:pairs", e)
            return False

    # ------------------ Load RAG (corpus) batches ------------------
    print("[step] Loading Corpus (RAG) batches")
    corpus_data = load_batches(
        args.corpus_chroma, args.collection_prefix, args.max_docs,
        workers=args.jobs if args.jobs > 0 else None,
        cache_dir=effective_cache_dir
    )
    if not corpus_data:
        raise RuntimeError("No RAG batches found under --corpus-chroma with the given --collection-prefix.")
    rag_embs = np.vstack([d["embeddings"] for d in corpus_data.values()])
    corpus_docs = sum([d["documents"] for d in corpus_data.values()], [])
    if rag_embs.size == 0 or len(corpus_docs) == 0:
        raise RuntimeError("Loaded RAG store but found no embeddings or documents.")

    # ------------------ Load/Build GENERIC ------------------
    generic_embs, generic_docs, generic_source = None, None, None

    if args.generic_chroma:
        print("[step] Loading Generic corpus (Chroma)")
        gen_data = load_chroma_embeddings(args.generic_chroma, None, args.max_docs, cache_dir=effective_cache_dir)
        if gen_data:
            generic_embs = np.vstack([d["embeddings"] for d in gen_data.values()])
            generic_docs = sum([d["documents"] for d in gen_data.values()], [])
            generic_source = "chroma"
        else:
            print("[warn] --generic-chroma provided but no collections found; falling back to --generic-model if available.")
    elif generic_embs is None and args.generic_model:
        source = preflight_model_check(args.generic_model, token=args.hf_token)
        print(f"[step] Embedding corpus docs with generic model ({source})")
        cache_path = os.path.join(cache_root, "corpus_generic_embeddings.npz")
        generic_embs = embed_with_finetuned(
            args.generic_model, corpus_docs, args.embed_limit, cache_path,
            force=args.force_embed, token=args.hf_token
        )
        generic_docs = corpus_docs
        generic_source = source

    # ------------------ Load FINETUNED ------------------
    finetuned_embs = None
    if args.finetuned_model:
        source = preflight_model_check(args.finetuned_model, token=args.hf_token, finetuned_base_model=args.finetuned_base_model)
        print(f"[step] Embedding corpus docs with fine-tuned model ({source})")
        cache_path = os.path.join(cache_root, "corpus_finetuned_embeddings.npz")
        finetuned_embs = embed_with_finetuned(
            args.finetuned_model, corpus_docs, args.embed_limit, cache_path,
            force=args.force_embed, token=args.hf_token, finetuned_base_model=args.finetuned_base_model
        )

    # ------------------ Assemble matrices & labels ------------------
    matrices: Dict[str, np.ndarray] = {"corpus_rag": rag_embs}
    matrix_labels: List[str] = ["corpus_rag"]

    if generic_embs is not None:
        matrices["generic"] = generic_embs
        matrix_labels.append("generic")

    if finetuned_embs is not None:
        matrices["finetuned"] = finetuned_embs
        matrix_labels.append("finetuned")

    # ------------------ Compute shift matrix (in-memory) ------------------
    shift_rows = []
    for a in tqdm(matrix_labels, desc="Shift matrix rows", unit="row"):
        row = {"": a}
        for b in tqdm(matrix_labels, desc=f"Comparing {a}", leave=False, unit="col"):
            try:
                sims = nearest_neighbor_stats(
                    matrices[a], matrices[b], args.k, args.samples, n_jobs=args.jobs,
                    align_mode=args.align_mode, aligned_dim=args.aligned_dim, proj_seed=args.proj_seed
                )
                row[b] = float(np.mean(sims)) if sims.size else float('nan')
            except Exception as e:
                safe_print(f"shift_compare:{a}_vs_{b}", e)
                row[b] = float('nan')
        shift_rows.append(row)

    # ------------------ Write embedding_shift_matrix.csv with completeness check ------------------
    final_csv = os.path.join(args.outdir, "embedding_shift_matrix.csv")
    required_cols = [""] + matrix_labels
    required_rows = matrix_labels

    try:
        if _csv_has_required_headers_and_rows(final_csv, required_cols, required_rows, row_label_col=""):
            print(f"[skip exists complete] {final_csv}")
        else:
            tmp_csv = final_csv + ".tmp"
            with open(tmp_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=required_cols)
                writer.writeheader()
                for row in shift_rows:
                    writer.writerow(row)
                    f.flush()
            os.replace(tmp_csv, final_csv)
            print(f"[written] {final_csv}")
    except Exception as e:
        safe_print("write_shift_matrix_csv", e)

    # ------------------ Compute additional drift metrics in-memory (skip if pairs CSV already complete) ------------------
    try:
        pairs_csv = os.path.join(args.outdir, "embedding_shift_pairs.csv")
        if _pairs_csv_complete(pairs_csv, matrix_labels):
            print(f"[skip exists complete] {pairs_csv}")
        else:
            # Build pairs from in-memory shift_rows without re-reading CSV
            shift_dict = {row[""]: {k: row[k] for k in matrix_labels} for row in shift_rows}

            pairs_data = []
            seen = set()
            for i, a in enumerate(matrix_labels):
                for j, b in enumerate(matrix_labels):
                    # Only emit unordered pairs once (including diagonal)
                    if j < i:
                        continue
                    val = shift_dict[a][b]
                    if val is None or (isinstance(val, float) and math.isnan(val)):
                        continue

                    self_sim_a = shift_dict[a][a]
                    drift_pct = 0.0 if a == b else ((self_sim_a - val) / self_sim_a) if self_sim_a else float('nan')

                    key = tuple(sorted((a, b)))
                    if key in seen:
                        continue
                    seen.add(key)

                    pairs_data.append({
                        "A": a,
                        "B": b,
                        "mean_cosine": float(val),
                        "relative_drift_pct": float(drift_pct * 100.0) if drift_pct is not None else float('nan'),
                    })

            # Write pairs CSV atomically
            tmp_pairs = pairs_csv + ".tmp"
            with open(tmp_pairs, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["A", "B", "mean_cosine", "relative_drift_pct"])
                writer.writeheader()
                writer.writerows(pairs_data)
                f.flush()
            os.replace(tmp_pairs, pairs_csv)
            print(f"[written] {pairs_csv}")
    except Exception as e:
        safe_print("compute_additional_drift_metrics", e)

    # ------------------ Plots: generate overlays for any pairs ------------------
    if args.plots:
        pairs = []
        labels = list(matrices.keys())
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                pairs.append((f"{labels[i]} vs {labels[j]}", (matrices[labels[i]], matrices[labels[j]])))

        if pairs:
            global_pairs = {}
            for label, (Xa, Xb) in tqdm(pairs, desc="Global overlay sims", unit="pair"):
                sims = nearest_neighbor_stats(
                    Xa, Xb, args.k, args.samples, n_jobs=args.jobs,
                    align_mode=args.align_mode, aligned_dim=args.aligned_dim, proj_seed=args.proj_seed
                )
                if sims.size:
                    global_pairs[label] = sims
            if global_pairs:
                overlay_histogram(global_pairs, "Global Similarity Comparison",
                                  os.path.join(args.outdir, "global_overlay.png"))

        # Math-heavy subset (only if we have at least two matrices)
        if len(matrices) >= 2:
            math_idx_corpus = mathy_indices(corpus_docs, args.math_thresh, workers=args.jobs)

            math_mats: Dict[str, np.ndarray] = {}
            if math_idx_corpus:
                math_mats["corpus_rag_math"] = rag_embs[math_idx_corpus]
                if finetuned_embs is not None:
                    math_mats["finetuned_math"] = finetuned_embs[math_idx_corpus]
                if generic_embs is not None:
                    if generic_source == "model":
                        math_mats["generic_math"] = generic_embs[math_idx_corpus]
                    elif generic_source == "chroma":
                        if generic_docs:
                            gen_idx = [i for i, t in enumerate(generic_docs)
                                       if mathiness_score(t) >= args.math_thresh]
                            if gen_idx:
                                math_mats["generic_math"] = generic_embs[gen_idx]

            if len(math_mats) >= 2:
                math_pairs = {}
                mlabels = list(math_mats.keys())
                for i in range(len(mlabels)):
                    for j in range(i + 1, len(mlabels)):
                        sims = nearest_neighbor_stats(
                            math_mats[mlabels[i]], math_mats[mlabels[j]],
                            args.k, min(args.samples, len(math_mats[mlabels[i]])),
                            n_jobs=args.jobs, align_mode=args.align_mode,
                            aligned_dim=args.aligned_dim, proj_seed=args.proj_seed
                        )
                        if sims.size:
                            math_pairs[f"{mlabels[i]} vs {mlabels[j]}"] = sims
                if math_pairs:
                    overlay_histogram(math_pairs, "Math-heavy Similarity Comparison",
                                      os.path.join(args.outdir, "math_overlay.png"))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        safe_print("main", e)

"""Construction and querying of a dense FAISS index.

The module prefers a multilingual SentenceTransformer model when available.
When the library or model cannot be loaded, it falls back to a lightweight
TF‑IDF + SVD (LSA) pipeline implemented with NumPy only.  In both cases the
produced vectors are L2 normalised and deterministic for a given input.

The command line interface and output file names remain backward compatible
with the previous dummy implementation.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import random
import re
import hashlib
from typing import Dict, Iterable, List, Tuple

import faiss
import numpy as np


# ---------------------------------------------------------------------------
# Determinism helpers
np.random.seed(0)
random.seed(0)
try:  # pragma: no cover - torch is optional
    import torch

    torch.manual_seed(0)
except Exception:  # pragma: no cover - keep going if torch isn't installed
    pass


ST_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
LSA_DIM = 256


# ---------------------------------------------------------------------------
# Tokenisation and simple TF‑IDF + SVD implementation
def _tokenise(text: str) -> List[str]:
    return re.findall(r"\b\w+\b", text.lower())


def _fit_lsa(texts: List[str], dim: int = LSA_DIM, *, normalize: bool = True) -> Tuple[np.ndarray, Dict]:
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import svds

    tokens = [_tokenise(t) for t in texts]

    vocab: Dict[str, int] = {}
    df: Dict[str, int] = {}
    for toks in tokens:
        seen = set()
        for tok in toks:
            if tok not in vocab:
                vocab[tok] = len(vocab)
            if tok not in seen:
                df[tok] = df.get(tok, 0) + 1
                seen.add(tok)

    n_docs = len(texts)
    vocab_size = len(vocab)
    idf = np.zeros(vocab_size, dtype=np.float32)
    for tok, idx in vocab.items():
        idf[idx] = np.log((n_docs + 1) / (df[tok] + 1)) + 1.0

    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    for i, toks in enumerate(tokens):
        counts: Dict[int, int] = {}
        for tok in toks:
            counts[vocab[tok]] = counts.get(vocab[tok], 0) + 1
        if counts:
            max_tf = max(counts.values())
            for idx, c in counts.items():
                tf = c / max_tf
                rows.append(i)
                cols.append(idx)
                data.append(tf * idf[idx])

    X = csr_matrix((data, (rows, cols)), shape=(n_docs, vocab_size), dtype=np.float32)

    dim = min(dim, min(X.shape))
    U, S, Vt = svds(X, k=dim)
    order = np.argsort(S)[::-1]
    U = U[:, order]
    S = S[order]
    Vt = Vt[order, :]

    for i in range(dim):  # deterministic sign
        if Vt[i, 0] < 0:
            U[:, i] *= -1
            Vt[i, :] *= -1

    X_red = U * S
    if normalize:
        X_red = X_red / (np.linalg.norm(X_red, axis=1, keepdims=True) + 1e-12)

    embed_info = {
        "dim": int(dim),
        "vocab": vocab,
        "idf": idf.tolist(),
        "vtd": base64.b64encode(pickle.dumps(Vt.T)).decode("utf-8"),
    }
    return X_red.astype("float32"), embed_info


def _lsa_embed(
    texts: Iterable[str],
    vocab: Dict[str, int],
    idf: List[float],
    vtd: str,
    *,
    normalize: bool = True,
) -> np.ndarray:
    Vt_T = pickle.loads(base64.b64decode(vtd))
    vocab_size = len(vocab)
    idf_vec = np.array(idf, dtype=np.float32)

    vecs = []
    for t in texts:
        toks = _tokenise(t)
        counts: Dict[int, int] = {}
        for tok in toks:
            idx = vocab.get(tok)
            if idx is not None:
                counts[idx] = counts.get(idx, 0) + 1

        vec = np.zeros(vocab_size, dtype=np.float32)
        if counts:
            max_tf = max(counts.values())
            for idx, c in counts.items():
                tf = c / max_tf
                vec[idx] = tf * idf_vec[idx]
        emb = vec @ Vt_T
        if normalize:
            emb = emb / (np.linalg.norm(emb) + 1e-12)
        vecs.append(emb.astype("float32"))

    return np.vstack(vecs)


# ---------------------------------------------------------------------------
# Index construction and search
def build_faiss_index(
    pages_jsonl: str,
    out_index_path: str,
    out_meta_path: str,
    cfg: Dict,
) -> Tuple[str, str]:
    texts: List[str] = []
    meta: List[Dict[str, str]] = []
    with open(pages_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            texts.append(rec["text"])
            meta.append(
                {
                    "doc_id": rec["doc_id"],
                    "page": rec["page"],
                    "chunk_id": rec.get("chunk_id", 0),
                }
            )

    backend = cfg.get("dense_backend", "auto")
    model_name = cfg.get("dense_model_name", ST_MODEL_NAME)
    device_opt = cfg.get("dense_device", "auto")
    max_batch = int(cfg.get("dense_max_batch", 64))
    normalize = bool(cfg.get("dense_normalize", True))

    device = "cpu"
    if device_opt == "cuda":
        device = "cuda"
    elif device_opt == "auto":
        try:  # pragma: no cover - optional
            if "torch" in globals() and torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass

    X = None
    embed_meta: Dict = {}
    if backend in ("auto", "sbert"):
        try:  # pragma: no cover - optional
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name, device=device)
            model.eval()
            parts = []
            for i in range(0, len(texts), max_batch):
                parts.append(
                    model.encode(
                        texts[i : i + max_batch],
                        batch_size=max_batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=normalize,
                    ).astype("float32")
                )
            X = np.vstack(parts)
            embed_meta = {
                "method": "sbert",
                "model": model_name,
                "normalize": normalize,
            }
        except Exception:
            X = None

    if X is None:
        X, lsa_info = _fit_lsa(texts, LSA_DIM, normalize=normalize)
        embed_meta = {"method": "lsa", "normalize": normalize, **lsa_info}

    dim = X.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(X)
    faiss.write_index(index, out_index_path)

    with open(out_meta_path, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "embedding": embed_meta}, f, ensure_ascii=False, indent=2)

    try:
        os.makedirs("/tmp", exist_ok=True)
        norms = np.linalg.norm(X, axis=1)
        with open("/tmp/dense.log", "w", encoding="utf-8") as lf:
            lf.write(
                "ntotal=%d dim=%d norm_mean=%.6f norm_min=%.6f norm_max=%.6f meta: method=%s model=%s normalize=%s\n"
                % (
                    index.ntotal,
                    dim,
                    float(norms.mean()),
                    float(norms.min()),
                    float(norms.max()),
                    embed_meta.get("method"),
                    embed_meta.get("model"),
                    embed_meta.get("normalize"),
                )
            )
    except Exception:
        pass

    return out_index_path, out_meta_path


def _load_meta(meta_path: str) -> Tuple[List[Dict], Dict]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta_obj = json.load(f)
    if isinstance(meta_obj, dict):
        meta = meta_obj.get("meta", [])
        embed_info = meta_obj.get("embedding", {})
    else:  # backward compatibility with old meta format
        meta = meta_obj
        embed_info = {}
    return meta, embed_info


def _embed_query(query: str, embed_info: Dict) -> np.ndarray:
    method = embed_info.get("method")
    normalize = embed_info.get("normalize", True)
    if method == "sbert":
        try:  # pragma: no cover - optional dependency
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(embed_info.get("model", ST_MODEL_NAME), device="cpu")
            model.eval()
            return model.encode(
                [query],
                batch_size=1,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=normalize,
            ).astype("float32")
        except Exception:
            pass

    if method == "lsa":
        return _lsa_embed(
            [query],
            embed_info["vocab"],
            embed_info["idf"],
            embed_info["vtd"],
            normalize=normalize,
        )

    # Fallback to hash-based embedding for legacy indices
    h = np.frombuffer(hashlib.sha1(query.encode("utf-8")).digest(), dtype=np.uint8).astype("float32")
    if normalize:
        h = h / (np.linalg.norm(h) + 1e-12)
    return h.reshape(1, -1)


def search_faiss(index_path: str, meta_path: str, query: str, top_k: int = 10) -> List[Dict]:
    index = faiss.read_index(index_path)
    meta, embed_info = _load_meta(meta_path)
    qv = _embed_query(query, embed_info)
    D, I = index.search(qv, top_k)

    res = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        m = meta[int(idx)]
        res.append({**m, "score_dense": float(score)})
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--out_index", required=True)
    ap.add_argument("--out_meta", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_index), exist_ok=True)
    build_faiss_index(args.pages, args.out_index, args.out_meta, {})


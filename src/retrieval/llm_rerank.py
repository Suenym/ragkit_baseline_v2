import json
import os
import re
from typing import List, Dict, Any

try:
    import google.generativeai as genai
except Exception:  # pragma: no cover - optional dep
    genai = None


class GeminiReranker:
    """Rerank passages using Google Gemini models."""

    def __init__(self, model: str = "gemini-1.5-flash", max_chars: int = 600):
        if genai is None:
            raise ImportError("google-generativeai is not installed")
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model)
        self.max_chars = max_chars

    def rerank(self, query: str, hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not hits:
            return hits
        subset = hits[:top_k]
        passages = []
        for h in subset:
            preview = h.get("text", "")[: self.max_chars]
            passages.append(f"{h['id']}: {preview}")
        prompt = (
            "Rank the following passages by relevance to the query.\n"
            f"Query: {query}\nPassages:\n" + "\n".join(passages) +
            "\nRespond with JSON {\"scores\":[{\"id\":int,\"score\":float},...]}"
        )
        try:
            resp = self.model.generate_content(prompt)
            text = getattr(resp, "text", str(resp))
            m = re.search(r"{.*}", text, re.S)
            data = json.loads(m.group(0)) if m else {}
            scores = {int(item["id"]): float(item["score"]) for item in data.get("scores", [])}
            for h in subset:
                h["rerank_score"] = scores.get(h["id"], 0.0)
            subset.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception:
            pass  # fallback to original order
        return subset + hits[top_k:]

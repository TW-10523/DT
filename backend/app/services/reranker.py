# app/services/reranker_service.py
from typing import List, Dict, Any
from app.core.config import settings
from app.services.ollama_client import chat, generate, OllamaError
import json
import logging

logger = logging.getLogger(__name__)

class RerankError(Exception):
    pass

async def rerank_candidates(query: str, candidates: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Rerank candidate chunks. Each candidate: {"id":..., "text": "...", "metadata": {...}}
    Returns top_k candidates with a 'score' field (higher is better).
    Implementation strategy:
      - Compose a prompt asking the support model to score each candidate 0-1 for relevance.
      - Parse model output expecting JSON: [{"id": "...", "score": 0.8}, ...]
    Note: You can replace with a learned reranker later.
    """
    if not candidates:
        return []

    # Build prompt with context — be concise to avoid huge prompts
    prompt_parts = [
        "You are a relevance reranker. Score each candidate for the query on a scale 0.0-1.0.",
        f"Query: {query}",
        "Candidates:"
    ]
    for c in candidates:
        snippet = c.get("text", "").replace("\n", " ")
        prompt_parts.append(f"- id: {c.get('id')}\n  text: {snippet[:800]}")  # limit snippet length

    prompt_parts.append("Return a JSON array of objects with fields: id, score (0.0-1.0).")
    prompt = "\n".join(prompt_parts)

    model = settings.SUPPORT_LLM_NAME if getattr(settings, "SUPPORT_LLM_NAME", None) else settings.SUPPORT_LLM_NAME
    try:
        # Try chat endpoint first (structured response likely)
        try:
            resp = await chat(model=model, messages=[{"role": "user", "content": prompt}], max_tokens=512, temperature=settings.SUPPORT_LLM_TEMPERATURE)
            # Try to extract text from possible shapes
            text = ""
            if isinstance(resp, dict):
                # Ollama chat may return {"choices":[{"message":{"content": "..."}}]}
                if "choices" in resp and resp["choices"]:
                    c0 = resp["choices"][0]
                    # adapt to varying structures
                    text = c0.get("message", {}).get("content") or c0.get("text") or resp.get("text", "")
                else:
                    text = resp.get("text", "")
            else:
                text = str(resp)
        except OllamaError:
            # Fallback to generate
            gen = await generate(model=model, prompt=prompt, max_tokens=512, temperature=settings.SUPPORT_LLM_TEMPERATURE)
            if isinstance(gen, dict):
                text = gen.get("text") or (gen.get("choices") and gen["choices"][0].get("text")) or ""
            else:
                text = str(gen)

        # attempt to parse JSON in output
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                # map scores back to candidates
                id_to_score = {str(p["id"]): float(p["score"]) for p in parsed if "id" in p and "score" in p}
                for c in candidates:
                    c["score"] = id_to_score.get(str(c.get("id")), 0.0)
                # sort by score desc
                sorted_c = sorted(candidates, key=lambda x: x.get("score", 0.0), reverse=True)
                return sorted_c[:top_k]
        except Exception:
            # If parsing failed, fallback to naive heuristic: ask model to return newline "id:score" lines
            pass

        # Fallback: simple heuristic using model scoring by prompting per-candidate (expensive)
        scored = []
        for c in candidates:
            short_prompt = f"Query: {query}\nCandidate: {c.get('text','')[:800]}\nScore relevance 0.0-1.0:"
            try:
                gen = await generate(model=model, prompt=short_prompt, max_tokens=8, temperature=0.0)
                text_out = gen.get("text") if isinstance(gen, dict) else str(gen)
                score = float(text_out.strip().split()[0])
            except Exception:
                score = 0.0
            c["score"] = score
            scored.append(c)
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

    except OllamaError as e:
        logger.error("Reranker Ollama error: %s", e)
        raise RerankError(str(e)) from e
    except Exception as e:
        logger.exception("Reranker unexpected error")
        raise RerankError(str(e)) from e

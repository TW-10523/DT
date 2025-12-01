# app/services/rag_service.py
from typing import List, Dict, Any, Optional
import logging
from app.core.config import settings
from app.repositories.postgres_repository import PostgresRepository  # adapt to your repo
from app.services.embedder import get_embeddings, EmbeddingError
from app.services.reranker import rerank_candidates, RerankError
from app.services.ollama_client import generate, chat, OllamaError

logger = logging.getLogger(__name__)

class RAGError(Exception):
    pass

async def answer_query(
    query: str,
    collection_name: Optional[str] = None,
    n_results: int = None,
    repo: Optional[PostgresRepository] = None,
    user_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    High-level RAG pipeline:
      1. If available, get embedding for the query
      2. Use repository to retrieve candidate chunks
      3. Rerank candidates with support model
      4. Build prompt and call primary model for final answer
      5. Return structured response
    """
    n_results = n_results or settings.SEARCH_DEFAULT_RESULTS
    try:
        # 1) Optionally compute query embedding (if repo search supports vector search)
        query_embedding = None
        if getattr(settings, "EMBEDDING_MODEL", None):
            try:
                vecs = await get_embeddings([query])
                if vecs and isinstance(vecs, list):
                    query_embedding = vecs[0]
            except EmbeddingError:
                logger.warning("Failed to compute embeddings for query; proceeding with textual retrieval")

        # 2) Retrieve candidates
        if repo is None:
            repo = PostgresRepository()  # instantiate your repo; ensure constructor signature matches
        candidates = await repo.search(collection=collection_name, query=query, embedding=query_embedding, top_k=max(n_results * 5, 20))
        # candidates: list of {"id":..., "text":..., "metadata": {...}}

        if not candidates:
            return {
                "answer_lines": ["No results found", "", "", ""],
                "metadata": {"confidence": 0.0, "sources": []},
                "formatted_response": "No results found"
            }

        # 3) Rerank candidates to top-K
        try:
            reranked = await rerank_candidates(query=query, candidates=candidates, top_k=n_results)
        except RerankError:
            # fallback to first n_results
            reranked = candidates[:n_results]

        # 4) Build generation prompt
        # Combine top contexts into prompt (keep lengths small)
        context_texts = []
        sources = []
        for c in reranked:
            snippet = c.get("text", "")
            context_texts.append(f"Source[{c.get('id')}]: {snippet[:1500]}")
            sources.append({"id": c.get("id"), "meta": c.get("metadata", {})})

        context_section = "\n\n".join(context_texts)
        prompt_template = (
            "You are an HR assistant. Use the following sources to answer the user query.\n\n"
            "Sources:\n"
            f"{context_section}\n\n"
            "Instructions:\n"
            "- Answer concisely in 4 lines.\n"
            "- Include a confidence score (0.0-1.0) in metadata.\n"
            "- If unsure, say you are unsure and provide recommendations.\n\n"
            f"User query: {query}\n\n"
            "Answer:"
        )

        # 5) Call primary model
        try:
            resp = await chat(
                model=settings.PRIMARY_LLM_NAME,
                messages=[{"role": "system", "content": "You are a helpful HR assistant."},
                          {"role": "user", "content": prompt_template}],
                max_tokens=settings.PRIMARY_LLM_MAX_TOKENS,
                temperature=settings.PRIMARY_LLM_TEMPERATURE
            )
            # Parse response — Ollama shapes vary; try common spots
            answer_text = ""
            if isinstance(resp, dict) and "choices" in resp and resp["choices"]:
                answer_text = resp["choices"][0].get("message", {}).get("content") or resp["choices"][0].get("text") or ""
            elif isinstance(resp, dict) and "text" in resp:
                answer_text = resp.get("text", "")
            else:
                answer_text = str(resp)
        except OllamaError:
            # fallback to generate
            gen = await generate(model=settings.PRIMARY_LLM_NAME, prompt=prompt_template, max_tokens=settings.PRIMARY_LLM_MAX_TOKENS)
            answer_text = gen.get("text") if isinstance(gen, dict) else str(gen)

        # Format answer lines: split into up to HR_ANSWER_LINES
        lines = [line.strip() for line in answer_text.strip().splitlines() if line.strip()]
        # ensure exactly settings.HR_ANSWER_LINES elements
        desired = settings.HR_ANSWER_LINES if hasattr(settings, "HR_ANSWER_LINES") else 4
        if len(lines) < desired:
            # pad with blank lines
            lines += [""] * (desired - len(lines))
        else:
            lines = lines[:desired]

        # compute naive confidence: try to parse a numeric from answer_text or set high if model used sources
        confidence = 0.9 if "I am not sure" not in answer_text.lower() else 0.4

        return {
            "answer_lines": lines,
            "metadata": {
                "confidence": confidence,
                "sources": sources
            },
            "formatted_response": answer_text
        }

    except Exception as e:
        logger.exception("RAG pipeline error")
        raise RAGError(str(e)) from e

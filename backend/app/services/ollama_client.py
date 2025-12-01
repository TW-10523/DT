# app/services/ollama_client.py
import httpx
from typing import Any, Dict, List, Optional
from app.core.config import settings
from asyncio import TimeoutError

OLLAMA_BASE = settings.PRIMARY_LLM_ENDPOINT.rstrip("/")  # e.g. http://localhost:11434

class OllamaError(Exception):
    pass

async def _build_client() -> httpx.AsyncClient:
    headers = {"Content-Type": "application/json"}
    if getattr(settings, "OLLAMA_API_KEY", None):
        headers["Authorization"] = f"Bearer {settings.OLLAMA_API_KEY}"
    timeout = httpx.Timeout(settings.OLLAMA_TIMEOUT_SECONDS if hasattr(settings, "OLLAMA_TIMEOUT_SECONDS") else 60)
    return httpx.AsyncClient(base_url=OLLAMA_BASE, headers=headers, timeout=timeout)

async def generate(
    model: str,
    prompt: str,
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    stop: Optional[List[str]] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Non-streaming generation call to Ollama.
    NOTE: Adjust path if your Ollama uses a different API (some versions use /api/generate or /v1/generate).
    """
    payload = {"model": model, "prompt": prompt}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    if stop:
        payload["stop"] = stop
    payload.update(kwargs)

    client = await _build_client()
    try:
        # Common Ollama path — change if needed
        resp = await client.post("/api/generate", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        text = e.response.text if e.response is not None else str(e)
        raise OllamaError(f"Generation failed: {text}") from e
    except (httpx.RequestError, TimeoutError) as e:
        raise OllamaError(f"Ollama request error: {str(e)}") from e
    finally:
        await client.aclose()

async def chat(
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
    temperature: Optional[float] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Chat-style completion (if Ollama exposes /api/chat/completions).
    """
    payload = {"model": model, "messages": messages}
    if max_tokens:
        payload["max_tokens"] = max_tokens
    if temperature is not None:
        payload["temperature"] = temperature
    payload.update(kwargs)

    client = await _build_client()
    try:
        resp = await client.post("/api/chat/completions", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        text = e.response.text if e.response is not None else str(e)
        raise OllamaError(f"Chat completion failed: {text}") from e
    except (httpx.RequestError, TimeoutError) as e:
        raise OllamaError(f"Ollama request error: {str(e)}") from e
    finally:
        await client.aclose()

async def embeddings(model: str, inputs: List[str]) -> Dict[str, Any]:
    """
    Request embeddings from Ollama (if supported).
    If your Ollama build doesn't provide embeddings, this should be replaced by a call to
    a dedicated embedding server or external embedding provider.
    """
    payload = {"model": model, "input": inputs}
    client = await _build_client()
    try:
        resp = await client.post("/api/embeddings", json=payload)
        resp.raise_for_status()
        return resp.json()
    except httpx.HTTPStatusError as e:
        raise OllamaError(f"Embeddings failed: {e.response.text}") from e
    except (httpx.RequestError, TimeoutError) as e:
        raise OllamaError(f"Ollama request error: {str(e)}") from e
    finally:
        await client.aclose()

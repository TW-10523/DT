# utils/search_pg.py
from core.logging import logger
from services.embedder import process_text, embed_text
from repositories.postgres_repository import PostgresRepository

repo = PostgresRepository()

def search_query_pg(collection_name: str, query_text: str, top_k: int = 10):
    try:
        cleaned = process_text(query_text)
        logger.info(f"[SEARCH] Processed Query: '{cleaned}'")
        vector = embed_text(cleaned)  # returns list/np array of floats
        # ensure vector is python list
        vlist = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        results = repo.nearest_neighbors(vlist, collection_name=collection_name, top_k=top_k)
        # results is list of {"chunk_id","distance","text","metadata"}
        # Convert to same shape as previous chroma result if other code expects that form
        documents = [r["text"] for r in results]
        metadatas = [r.get("metadata", {}) for r in results]
        distances = [r["distance"] for r in results]
        return {"documents": [documents], "metadatas": [metadatas], "distances": [distances]}
    except Exception as e:
        logger.error(f"[SEARCH_QUERY_PG] Failed query: {e}", exc_info=True)
        return {"documents": [[]]}

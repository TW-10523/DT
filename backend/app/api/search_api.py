# backend/api/search_api.py
# NEW: RAG query endpoints that integrate with Postgres + pgvector and your LLM service
# REUSED: search / reranking logic from aviary rag_service, reranker_service, embedder
from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from services.rag_service import search_rag  # REUSED/adapted
from core.db import get_db  # NEW: DB session

router = APIRouter()

class SearchRequest(BaseModel):
    collection_name: str | None = None
    query: str
    n_results: int = 3
    language: str | None = None

class SearchResponse(BaseModel):
    results: list

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db=Depends(get_db)):
    """
    Perform RAG: embed query -> run pgvector similarity -> rerank -> send to LLM
    REUSED: search_rag implementation (adapt to use Postgres instead of Chroma)
    """
    try:
        return search_rag(req, db=db)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class DeleteRequest(BaseModel):
    document_id: int

@router.delete("/document")
def delete_document(req: DeleteRequest, db=Depends(get_db)):
    # NEW: delete from Postgres tables; reuse logic from aviary record_service where possible
    from services.record_service import delete_document
    try:
        delete_document(req.document_id, db=db)
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

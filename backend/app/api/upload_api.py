# backend/api/upload_api.py
# NEW file: upload REST endpoints for file ingestion
# REUSED: high-level ingestion logic (extract -> chunk -> embed -> store) from aviary document_service/embedder

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from pydantic import BaseModel
from app.core.db import get_async_db  # NEW: DB dependency for Postgres

router = APIRouter()

class UploadResponse(BaseModel):
    document_id: int
    message: str

@router.post("/", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...), db=Depends(get_async_db)):
    """
    Upload a PDF/DOCX/TXT -> extract text -> chunk -> embed -> store in Postgres+pgvector.
    REUSE: document text extraction + embedding functions from aviary (services.document_service)
    NEW: storing into Postgres tables (models) instead of Chroma
    """
    try:
        doc_id = await ingest_document(file.filename, await file.read(), db=db)
        return {"document_id": doc_id, "message": "Ingested successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

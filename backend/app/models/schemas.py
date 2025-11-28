# backend/models/schemas.py
# NEW: pydantic schemas used by API endpoints (adapted from aviary models.schemas)
from pydantic import BaseModel
from typing import List, Optional

class ChunkOut(BaseModel):
    id: int
    text: str
    class Config:
        orm_mode = True

class DocumentOut(BaseModel):
    id: int
    filename: str
    title: Optional[str]
    chunks: List[ChunkOut] = []
    class Config:
        orm_mode = True

class SearchRequest(BaseModel):
    query: str
    n_results: int = 3
    language: Optional[str] = None

class DeleteRequest(BaseModel):
    document_id: int

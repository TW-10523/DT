# backend/api/search_api.py
# NEW: RAG query endpoints that integrate with Postgres + pgvector and your LLM service
# REUSED: search / reranking logic from aviary rag_service, reranker_service, embedder
from fastapi import APIRouter, HTTPException, Body, Depends
from pydantic import BaseModel
from typing import Optional
from app.services.hr_assistant_service import HRAssistantService
from app.services.translation_service import EnhancedTranslationService
from app.models.hr_assistant_model import (
    HRQueryRequest,
    HRQueryResponse,
    HRAssistantError,
    HRCollectionsResponse,
    HRDocumentUploadRequest,
    HRDocumentUploadResponse,
    HRFeedbackRequest,
    HRFeedbackResponse
)
from app.core.db import get_async_db
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class SearchRequest(BaseModel):
    collection_name: str | None = None
    query: str
    n_results: int = 3
    language: str | None = None

class SearchResponse(BaseModel):
    results: list

@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest, db=Depends(get_async_db)):
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
def delete_document(req: DeleteRequest, db=Depends(get_async_db)):
    # NEW: delete from Postgres tables; reuse logic from aviary record_service where possible
    from services.record_service import delete_document
    try:
        delete_document(req.document_id, db=db)
        return {"message": "deleted"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# HR Assistant endpoints
@router.post("/hr/query", response_model=HRQueryResponse)
def hr_assistant_query(req: HRQueryRequest, db=Depends(get_async_db)):
    """
    Process HR query through retrieval-augmented generation pipeline.
    Returns structured 4-line answer with metadata.
    """
    try:
        # Initialize services
        from services.rag_service import search_rag
        from services.inference import InferenceService
        
        # Create translation service
        translator = EnhancedTranslationService(use_cache=True)
        
        # Create mock retriever that uses existing search_rag
        class RetrieverAdapter:
            def search(self, query, collection_name, n_results, db):
                req = SearchRequest(
                    query=query,
                    collection_name=collection_name,
                    n_results=n_results
                )
                result = search_rag(req, db=db)
                return result.get('results', [])
        
        # Create LLM service adapter
        class LLMAdapter:
            def __init__(self):
                self.inference_service = InferenceService()
            
            def generate(self, prompt, temperature=0.0, max_tokens=500):
                # Adapt to use existing inference service
                return self.inference_service.generate_text(
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
        
        # Initialize HR Assistant
        retriever = RetrieverAdapter()
        llm = LLMAdapter()
        hr_assistant = HRAssistantService(
            retriever_service=retriever,
            translator_service=translator,
            llm_service=llm
        )
        
        # Process query
        result = hr_assistant.process_query(
            query=req.query,
            collection_name=req.collection_name,
            n_results=req.n_results,
            db=db
        )
        
        # Format response according to model
        return HRQueryResponse(
            answer_lines=result['answer_lines'],
            metadata=result['metadata'],
            formatted_response=result['formatted_response']
        )
        
    except Exception as e:
        logger.error(f"HR Assistant error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=HRAssistantError(
                error=str(e),
                error_type="processing_error",
                details={"query": req.query}
            ).dict()
        )

@router.get("/hr/collections", response_model=HRCollectionsResponse)
def list_hr_collections(db=Depends(get_async_db)):
    """
    List available HR document collections.
    """
    try:
        # Placeholder implementation - would query actual collections
        collections = [
            {
                "collection_name": "hr_policies",
                "description": "Company HR policies and procedures",
                "document_count": 45,
                "languages": ["en", "ja"],
                "last_updated": "2024-01-15T10:00:00Z"
            },
            {
                "collection_name": "benefits",
                "description": "Employee benefits documentation",
                "document_count": 23,
                "languages": ["en"],
                "last_updated": "2024-01-10T08:30:00Z"
            },
            {
                "collection_name": "leave_policies",
                "description": "Leave and time-off policies",
                "document_count": 12,
                "languages": ["en", "ja"],
                "last_updated": "2024-01-20T14:15:00Z"
            }
        ]
        
        total_docs = sum(c["document_count"] for c in collections)
        
        return HRCollectionsResponse(
            collections=collections,
            total_documents=total_docs
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hr/document", response_model=HRDocumentUploadResponse)
def upload_hr_document(req: HRDocumentUploadRequest, db=Depends(get_async_db)):
    """
    Upload a new HR document to the specified collection.
    """
    try:
        # Placeholder implementation
        # In production, would:
        # 1. Validate collection exists
        # 2. Process document (chunk, embed)
        # 3. Store in database
        
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Simulate chunking
        chunk_size = 500
        chunks_created = max(1, len(req.document_content) // chunk_size)
        
        return HRDocumentUploadResponse(
            doc_id=doc_id,
            status="success",
            message=f"Document '{req.document_title}' uploaded successfully",
            chunks_created=chunks_created
        )
        
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/hr/feedback", response_model=HRFeedbackResponse)
def submit_feedback(req: HRFeedbackRequest, db=Depends(get_async_db)):
    """
    Submit user feedback on HR assistant response.
    """
    try:
        import uuid
        feedback_id = str(uuid.uuid4())
        
        # In production, would store feedback in database
        # for continuous improvement
        
        return HRFeedbackResponse(
            feedback_id=feedback_id,
            status="received",
            message="Thank you for your feedback. It will help us improve the HR assistant."
        )
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

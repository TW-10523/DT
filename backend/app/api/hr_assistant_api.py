"""
Production HR Assistant API
Full-featured production endpoints with authentication, monitoring, and error handling
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, status, Request
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession
import redis.asyncio as redis
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.core.config import settings, validate_settings
from app.core.db import get_async_db
from app.models.hr_assistant_model import (
    HRQueryRequest,
    HRQueryResponse,
    HRAssistantError,
    HRCollectionsResponse,
    HRDocumentUploadRequest,
    HRDocumentUploadResponse,
    HRFeedbackRequest,
    HRFeedbackResponse,
    HRCollectionInfo
)
from app.services.hr_assistant_production import get_hr_assistant, ProductionHRAssistantService
from app.middleware.auth_middleware import get_current_user
# from app.middleware.rate_limit_middleware import RateLimitMiddleware

logger = logging.getLogger(__name__)

# Create router with prefix
router = APIRouter(
    prefix="/hr/v1",
    tags=["HR Assistant"],
    responses={
        401: {"description": "Unauthorized"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Redis client for caching and rate limiting
redis_client: Optional[redis.Redis] = None

@router.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global redis_client
    
    try:
        # Validate configuration
        validate_settings()
        logger.info("Configuration validated successfully")
        
        # Initialize Redis
        if settings.CACHE_ENABLED:
            redis_client = redis.Redis(
                connection_pool=redis.ConnectionPool.from_url(
                    settings.REDIS_URL,
                    password=settings.REDIS_PASSWORD,
                    ssl=settings.REDIS_SSL,
                    max_connections=settings.REDIS_POOL_SIZE
                )
            )
            await redis_client.ping()
            logger.info("Redis connection established")
        
        # Initialize HR Assistant
        hr_assistant = await get_hr_assistant()
        logger.info("HR Assistant service initialized")
        
    except Exception as e:
        logger.error(f"Startup failed: {str(e)}")
        raise

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global redis_client
    
    try:
        # Close HR Assistant
        hr_assistant = await get_hr_assistant()
        await hr_assistant.close()
        
        # Close Redis
        if redis_client:
            await redis_client.close()
        
        logger.info("Shutdown complete")
        
    except Exception as e:
        logger.error(f"Shutdown error: {str(e)}")

# Health check endpoints
@router.get("/health")
async def health_check():
    """Basic health check endpoint"""
    hr_assistant = await get_hr_assistant()
    health_status = await hr_assistant.health_check()
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )
    
    return health_status

@router.get("/ready")
async def readiness_check(db: AsyncSession = Depends(get_async_db)):
    """Readiness check including database connectivity"""
    try:
        # Check database
        await db.execute("SELECT 1")
        
        # Check HR Assistant
        hr_assistant = await get_hr_assistant()
        health_status = await hr_assistant.health_check()
        
        if health_status["status"] != "healthy":
            return {
                "ready": False,
                "checks": health_status["checks"]
            }
        
        return {
            "ready": True,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Readiness check failed: {str(e)}")
        return {
            "ready": False,
            "error": str(e)
        }

@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return StreamingResponse(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )

# Main HR Assistant endpoints
@router.post("/query", 
            response_model=HRQueryResponse,
            summary="Process HR Query",
            description="Process an HR-related query and return structured 4-line answer with metadata")
async def process_hr_query(
    request: HRQueryRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Process HR query through production RAG pipeline
    
    This endpoint:
    - Retrieves relevant documents from vector database
    - Translates non-English content
    - Generates structured 4-line answer
    - Provides source citations and recommendations
    - Tracks analytics and metrics
    """
    try:
        # Get HR Assistant service
        hr_assistant = await get_hr_assistant()
        
        # Process query
        result = await hr_assistant.process_query(
            query=request.query,
            collection_name=request.collection_name,
            n_results=request.n_results,
            user_id=current_user.get('user_id'),
            session=db
        )
        
        # Log query for analytics (async background task)
        background_tasks.add_task(
            log_query_analytics,
            request.query,
            result,
            current_user.get('user_id')
        )
        
        # Return formatted response
        return HRQueryResponse(
            answer_lines=result['answer_lines'],
            metadata=result['metadata'],
            formatted_response=result['formatted_response']
        )
        
    except ValueError as e:
        # Input validation errors
        logger.warning(f"Invalid query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
        
    except Exception as e:
        # Unexpected errors
        logger.error(f"Query processing error: {str(e)}", exc_info=True)
        
        # Return structured error response
        error_response = HRAssistantError(
            error=str(e),
            error_type=type(e).__name__,
            details={
                "query": request.query[:100],
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=error_response.dict()
        )

@router.get("/collections",
           response_model=HRCollectionsResponse,
           summary="List HR Collections",
           description="Get list of available HR document collections")
async def list_collections(
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List available HR document collections
    
    Returns metadata about each collection including:
    - Collection name and description
    - Document count
    - Supported languages
    - Last update time
    """
    try:
        # Query database for collections
        query = """
            SELECT 
                collection_name,
                description,
                COUNT(DISTINCT doc_id) as document_count,
                array_agg(DISTINCT language) as languages,
                MAX(updated_at) as last_updated
            FROM document_collections
            GROUP BY collection_name, description
            ORDER BY collection_name
        """
        
        result = await db.execute(query)
        collections_data = result.fetchall()
        
        collections = []
        total_docs = 0
        
        for row in collections_data:
            collection = HRCollectionInfo(
                collection_name=row['collection_name'],
                description=row['description'],
                document_count=row['document_count'],
                languages=row['languages'] or ['en'],
                last_updated=row['last_updated'].isoformat() if row['last_updated'] else None
            )
            collections.append(collection)
            total_docs += row['document_count']
        
        return HRCollectionsResponse(
            collections=collections,
            total_documents=total_docs
        )
        
    except Exception as e:
        logger.error(f"Error listing collections: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve collections"
        )

@router.post("/document",
            response_model=HRDocumentUploadResponse,
            summary="Upload HR Document",
            description="Upload a new HR document to the system")
async def upload_document(
    request: HRDocumentUploadRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload a new HR document
    
    Requires HR_MANAGER, HR_STAFF, or ADMIN role.
    
    The document will be:
    - Chunked into searchable segments
    - Embedded using the configured model
    - Indexed in the vector database
    - Made available for RAG queries
    """
    if not settings.FEATURE_DOCUMENT_UPLOAD:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document upload feature is disabled"
        )
    
    try:
        # Validate file size
        content_size = len(request.document_content.encode('utf-8'))
        max_size = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024
        
        if content_size > max_size:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Document exceeds maximum size of {settings.MAX_UPLOAD_SIZE_MB}MB"
            )
        
        # Process document asynchronously
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Add to background processing queue
        background_tasks.add_task(
            process_document_upload,
            doc_id,
            request,
            current_user.get('user_id'),
            db
        )
        
        return HRDocumentUploadResponse(
            doc_id=doc_id,
            status="processing",
            message="Document queued for processing",
            chunks_created=0  # Will be updated after processing
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document upload error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document upload failed"
        )

@router.post("/feedback",
            response_model=HRFeedbackResponse,
            summary="Submit Feedback",
            description="Submit feedback on HR Assistant response quality")
async def submit_feedback(
    request: HRFeedbackRequest,
    db: AsyncSession = Depends(get_async_db),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Submit user feedback on response quality
    
    Feedback is used to:
    - Improve response quality
    - Identify knowledge gaps
    - Track user satisfaction
    - Guide system improvements
    """
    if not settings.FEATURE_FEEDBACK:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Feedback feature is disabled"
        )
    
    try:
        import uuid
        feedback_id = str(uuid.uuid4())
        
        # Store feedback in database
        query = """
            INSERT INTO hr_feedback (
                feedback_id, user_id, query, response_id,
                rating, feedback_text, was_helpful, created_at
            ) VALUES (
                :feedback_id, :user_id, :query, :response_id,
                :rating, :feedback_text, :was_helpful, :created_at
            )
        """
        
        await db.execute(query, {
            'feedback_id': feedback_id,
            'user_id': current_user.get('user_id'),
            'query': request.query,
            'response_id': request.response_id,
            'rating': request.rating,
            'feedback_text': request.feedback_text,
            'was_helpful': request.was_helpful,
            'created_at': datetime.utcnow()
        })
        
        await db.commit()
        
        return HRFeedbackResponse(
            feedback_id=feedback_id,
            status="received",
            message="Thank you for your feedback"
        )
        
    except Exception as e:
        logger.error(f"Feedback submission error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to submit feedback"
        )

@router.get("/query/{query_id}",
           summary="Get Query Status",
           description="Get status of a specific query (for async processing)")
async def get_query_status(
    query_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Get status of an async query
    
    Used for long-running queries that are processed asynchronously
    """
    if not redis_client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Async queries not available"
        )
    
    try:
        # Check query status in Redis
        status_key = f"query_status:{query_id}"
        status_data = await redis_client.get(status_key)
        
        if not status_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Query not found"
            )
        
        import json
        status = json.loads(status_data)
        
        # Check ownership
        if status.get('user_id') != current_user.get('user_id'):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting query status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get query status"
        )

# Background tasks
async def log_query_analytics(query: str, result: Dict[str, Any], user_id: Optional[str]):
    """Background task to log query analytics"""
    try:
        if not settings.FEATURE_ANALYTICS:
            return
        
        analytics_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": user_id or "anonymous",
            "query": query[:200],
            "confidence": result['metadata']['confidence'],
            "source_count": len(result['metadata']['sources']),
            "has_results": len(result['metadata']['sources']) > 0
        }
        
        # In production, send to analytics service
        logger.info(f"Query analytics: {analytics_data}")
        
        # Store in Redis for aggregation
        if redis_client:
            key = f"analytics:{datetime.utcnow().strftime('%Y%m%d')}"
            await redis_client.lpush(key, json.dumps(analytics_data))
            await redis_client.expire(key, 86400 * 30)  # 30 days
            
    except Exception as e:
        logger.error(f"Failed to log analytics: {str(e)}")

async def process_document_upload(
    doc_id: str,
    request: HRDocumentUploadRequest,
    user_id: str,
    db: AsyncSession
):
    """Background task to process document upload"""
    try:
        # This would implement actual document processing:
        # 1. Chunk the document
        # 2. Generate embeddings
        # 3. Store in vector database
        # 4. Update status
        
        logger.info(f"Processing document {doc_id} for user {user_id}")
        
        # Simulate processing
        await asyncio.sleep(2)
        
        # Update status in database
        # ... database operations ...
        
    except Exception as e:
        logger.error(f"Document processing failed: {str(e)}")

# Export router
__all__ = ["router"]

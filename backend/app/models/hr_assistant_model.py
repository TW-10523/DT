"""
Pydantic models for HR Assistant API requests and responses
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
import json


class HRQueryRequest(BaseModel):
    """Request model for HR assistant queries"""
    query: str = Field(..., min_length=1, max_length=500, description="User's HR-related question")
    collection_name: Optional[str] = Field(None, description="Optional collection to search in")
    n_results: int = Field(5, ge=1, le=20, description="Number of passages to retrieve")
    language: Optional[str] = Field(None, description="Preferred response language (default: en)")
    
    @validator('query')
    def validate_query(cls, v):
        """Ensure query is not just whitespace"""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()


class SourceDocument(BaseModel):
    """Model for source document metadata"""
    doc_id: str = Field(..., description="Document identifier")
    title: str = Field(..., description="Document title")
    page: int = Field(..., ge=0, description="Page number")
    original_snippet: str = Field(..., description="Original text snippet")
    translated_snippet: str = Field(..., description="Translated snippet (English)")
    score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class RecommendationItem(BaseModel):
    """Model for recommendation items"""
    title: str = Field(..., max_length=100, description="Recommendation title")
    reason: str = Field(..., max_length=200, description="Reason for recommendation")


class HRAssistantMetadata(BaseModel):
    """Metadata for HR assistant response"""
    sources: List[SourceDocument] = Field(default_factory=list, description="Source documents used")
    recommendations: List[RecommendationItem] = Field(
        default_factory=list, 
        max_items=3,
        description="Related query recommendations"
    )
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")


class HRQueryResponse(BaseModel):
    """Response model for HR assistant queries"""
    answer_lines: List[str] = Field(
        ..., 
        min_items=4, 
        max_items=4,
        description="Exactly 4 lines of answer text"
    )
    metadata: HRAssistantMetadata = Field(..., description="Response metadata")
    formatted_response: str = Field(..., description="Complete formatted response")
    
    @validator('answer_lines')
    def validate_answer_lines(cls, v):
        """Ensure exactly 4 non-empty lines"""
        if len(v) != 4:
            raise ValueError("Must have exactly 4 answer lines")
        # Ensure each line is meaningful (not just whitespace)
        for i, line in enumerate(v):
            if not line or len(line.strip()) == 0:
                v[i] = "Additional information not available."
        return v
    
    @validator('formatted_response')
    def validate_formatted_response(cls, v, values):
        """Ensure formatted response follows the specification"""
        if 'answer_lines' not in values or 'metadata' not in values:
            return v
        
        # Verify format: 4 lines + JSON metadata
        lines = v.strip().split('\n')
        if len(lines) < 5:  # At least 4 answer lines + 1 JSON line
            raise ValueError("Formatted response must contain 4 lines plus JSON metadata")
        
        # Verify last part is valid JSON
        try:
            # Find where JSON starts (after the 4th newline)
            json_start_idx = 0
            newline_count = 0
            for i, char in enumerate(v):
                if char == '\n':
                    newline_count += 1
                    if newline_count == 4:
                        json_start_idx = i + 1
                        break
            
            if json_start_idx > 0:
                json_str = v[json_start_idx:]
                json.loads(json_str)
        except (json.JSONDecodeError, IndexError):
            raise ValueError("Formatted response must end with valid JSON metadata")
        
        return v


class HRAssistantError(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_type: str = Field(..., description="Type of error")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class HRCollectionInfo(BaseModel):
    """Information about available HR document collections"""
    collection_name: str = Field(..., description="Collection identifier")
    description: str = Field(..., description="Collection description")
    document_count: int = Field(..., ge=0, description="Number of documents")
    languages: List[str] = Field(default_factory=list, description="Available languages")
    last_updated: Optional[str] = Field(None, description="Last update timestamp")


class HRCollectionsResponse(BaseModel):
    """Response for listing available collections"""
    collections: List[HRCollectionInfo] = Field(default_factory=list, description="Available collections")
    total_documents: int = Field(..., ge=0, description="Total documents across all collections")


class HRDocumentUploadRequest(BaseModel):
    """Request model for uploading HR documents"""
    collection_name: str = Field(..., description="Target collection")
    document_title: str = Field(..., max_length=200, description="Document title")
    document_content: str = Field(..., description="Document content")
    language: str = Field("en", description="Document language")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata")


class HRDocumentUploadResponse(BaseModel):
    """Response for document upload"""
    doc_id: str = Field(..., description="Assigned document ID")
    status: str = Field(..., description="Upload status")
    message: str = Field(..., description="Status message")
    chunks_created: int = Field(..., ge=0, description="Number of chunks created")


class HRFeedbackRequest(BaseModel):
    """Request model for user feedback on responses"""
    query: str = Field(..., description="Original query")
    response_id: Optional[str] = Field(None, description="Response identifier")
    rating: int = Field(..., ge=1, le=5, description="Rating (1-5)")
    feedback_text: Optional[str] = Field(None, max_length=1000, description="Optional feedback text")
    was_helpful: bool = Field(..., description="Whether response was helpful")


class HRFeedbackResponse(BaseModel):
    """Response for feedback submission"""
    feedback_id: str = Field(..., description="Feedback identifier")
    status: str = Field(..., description="Submission status")
    message: str = Field(..., description="Confirmation message")

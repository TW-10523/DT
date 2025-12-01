#!/usr/bin/env python3
"""
Minimal test API server for HR Assistant
Provides mock responses for testing
"""

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import json
import random

app = FastAPI(title="HR Assistant Test API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class HRQueryRequest(BaseModel):
    query: str
    collection_name: Optional[str] = None
    n_results: int = 5

class HRQueryResponse(BaseModel):
    answer_lines: List[str]
    metadata: Dict[str, Any]
    formatted_response: str

# Mock data
MOCK_RESPONSES = {
    "leave": {
        "answer_lines": [
            "Employees receive 15 days of paid annual leave after probation.",
            "Leave accrual starts from day one at 1.25 days per month.",
            "Unused leave can be carried forward up to 5 days.",
            "Part-time employees receive prorated leave based on hours."
        ],
        "sources": [
            {
                "doc_id": "HR-POL-001",
                "title": "Leave Policy 2024",
                "page": 3,
                "original_snippet": "Annual leave entitlement is 15 days",
                "translated_snippet": "Annual leave entitlement is 15 days",
                "score": 0.92
            }
        ],
        "confidence": 0.85
    },
    "benefits": {
        "answer_lines": [
            "Comprehensive health insurance covers medical, dental, and vision.",
            "Coverage begins on the first day of employment for full-time staff.",
            "Family members can be added with additional premium contributions.",
            "Annual wellness benefits include gym membership reimbursement."
        ],
        "sources": [
            {
                "doc_id": "HR-BEN-001",
                "title": "Health Benefits",
                "page": 1,
                "original_snippet": "Comprehensive health insurance coverage",
                "translated_snippet": "Comprehensive health insurance coverage",
                "score": 0.88
            }
        ],
        "confidence": 0.90
    },
    "default": {
        "answer_lines": [
            "No authoritative answer found in the docs.",
            "The query did not match any relevant HR documentation.",
            "Please try rephrasing your question or contact HR directly.",
            "You may also browse the HR portal for general information."
        ],
        "sources": [],
        "confidence": 0.0
    }
}

@app.get("/")
async def root():
    return {"service": "HR Assistant Test API", "status": "operational"}

@app.get("/hr/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "checks": {
            "database": "healthy",
            "redis": "healthy",
            "llm": "healthy"
        }
    }

@app.post("/hr/v1/query", response_model=HRQueryResponse)
async def process_query(
    request: HRQueryRequest,
    x_api_key: Optional[str] = Header(None)
):
    # Check API key
    if not x_api_key or x_api_key not in ["test-api-key-1", "test-api-key-2"]:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    # Determine response based on query
    query_lower = request.query.lower()
    
    if "leave" in query_lower or "vacation" in query_lower or "有給" in query_lower:
        response_data = MOCK_RESPONSES["leave"]
    elif "benefit" in query_lower or "insurance" in query_lower or "health" in query_lower:
        response_data = MOCK_RESPONSES["benefits"]
    elif "unicorn" in query_lower or "dragon" in query_lower:
        response_data = MOCK_RESPONSES["default"]
    else:
        # Random response for other queries
        response_data = random.choice([MOCK_RESPONSES["leave"], MOCK_RESPONSES["benefits"]])
        response_data["confidence"] *= 0.7  # Lower confidence for uncertain matches
    
    # Build recommendations
    recommendations = [
        {"title": "Check policy details", "reason": "View complete documentation"},
        {"title": "Contact HR", "reason": "Get personalized assistance"}
    ]
    
    # Build metadata
    metadata = {
        "sources": response_data["sources"],
        "recommendations": recommendations,
        "confidence": round(response_data["confidence"], 2)
    }
    
    # Build formatted response
    formatted_lines = response_data["answer_lines"]
    formatted_json = json.dumps(metadata, separators=(',', ':'))
    formatted_response = '\n'.join(formatted_lines) + '\n' + formatted_json
    
    return HRQueryResponse(
        answer_lines=response_data["answer_lines"],
        metadata=metadata,
        formatted_response=formatted_response
    )

@app.get("/hr/v1/collections")
async def list_collections(x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    return {
        "collections": [
            {
                "collection_name": "hr_policies",
                "description": "HR policies and procedures",
                "document_count": 15,
                "languages": ["en"],
                "last_updated": "2024-01-15T10:00:00Z"
            },
            {
                "collection_name": "benefits",
                "description": "Employee benefits documentation",
                "document_count": 8,
                "languages": ["en"],
                "last_updated": "2024-01-10T08:30:00Z"
            }
        ],
        "total_documents": 23
    }

@app.post("/hr/v1/feedback")
async def submit_feedback(request: dict, x_api_key: Optional[str] = Header(None)):
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
    
    return {
        "feedback_id": "test-feedback-123",
        "status": "received",
        "message": "Thank you for your feedback"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

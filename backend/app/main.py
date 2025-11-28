# backend/main.py
# REUSED: small FastAPI app pattern (from your aviary main.py)
# NEW: imports wired for new routers and PostgreSQL-backed RAG

from fastapi import FastAPI
from api.upload_api import router as upload_router
from api.search_api import router as search_router

app = FastAPI(docs_url="/docs")

# include routers
app.include_router(upload_router, prefix="/upload", tags=["upload"])
app.include_router(search_router, prefix="/rag", tags=["rag"])

@app.get("/health")
def health():
    return {"status": "ok"}

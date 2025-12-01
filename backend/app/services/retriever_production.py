"""
Production Retriever Service with Vector Database Integration
Handles efficient retrieval with connection pooling and caching
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, AsyncEngine
from sqlalchemy.pool import NullPool
from sqlalchemy import text
import asyncpg
from sentence_transformers import SentenceTransformer
from prometheus_client import Histogram

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
retrieval_duration = Histogram('retriever_query_duration_seconds', 'Retrieval query duration')
embedding_duration = Histogram('retriever_embedding_duration_seconds', 'Embedding generation duration')

class ProductionRetrieverService:
    """
    Production retriever with vector database integration
    """
    
    def __init__(self):
        self.engine: Optional[AsyncEngine] = None
        self.embedding_model: Optional[SentenceTransformer] = None
        self.connection_pool: Optional[asyncpg.Pool] = None
        
    async def initialize(self):
        """Initialize retriever service"""
        try:
            # Create async engine with connection pooling
            self.engine = create_async_engine(
                settings.VECTOR_DB_URL,
                pool_size=settings.DATABASE_POOL_SIZE,
                max_overflow=settings.DATABASE_MAX_OVERFLOW,
                pool_timeout=settings.DATABASE_POOL_TIMEOUT,
                pool_recycle=settings.DATABASE_POOL_RECYCLE,
                echo=settings.DATABASE_ECHO,
                pool_pre_ping=True,  # Verify connections before use
            )
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer(
                settings.EMBEDDING_MODEL,
                device=settings.EMBEDDING_DEVICE,
                cache_folder=settings.EMBEDDING_CACHE_DIR
            )
            
            # Create connection pool for pgvector operations
            self.connection_pool = await asyncpg.create_pool(
                settings.VECTOR_DB_URL.replace('postgresql://', ''),
                min_size=5,
                max_size=settings.DATABASE_POOL_SIZE,
                command_timeout=settings.SEARCH_TIMEOUT_SECONDS
            )
            
            logger.info("Retriever service initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize retriever: {str(e)}")
            raise
    
    async def close(self):
        """Cleanup resources"""
        if self.engine:
            await self.engine.dispose()
        
        if self.connection_pool:
            await self.connection_pool.close()
    
    async def search_async(self,
                          query: str,
                          collection_name: Optional[str] = None,
                          n_results: int = 5,
                          min_score: float = 0.5,
                          session: Optional[AsyncSession] = None) -> List[Dict[str, Any]]:
        """
        Async vector similarity search
        
        Args:
            query: Search query
            collection_name: Collection to search
            n_results: Number of results
            min_score: Minimum similarity score
            session: Database session
            
        Returns:
            List of retrieved documents
        """
        with retrieval_duration.time():
            try:
                # Generate embedding
                embedding = await self._generate_embedding_async(query)
                
                # Perform vector search
                results = await self._vector_search(
                    embedding, 
                    collection_name, 
                    n_results, 
                    min_score
                )
                
                # Rerank if enabled
                if settings.RERANK_ENABLED:
                    results = await self._rerank_results(query, results)
                
                return results
                
            except Exception as e:
                logger.error(f"Search error: {str(e)}")
                raise
    
    async def _generate_embedding_async(self, text: str) -> np.ndarray:
        """Generate embedding for text"""
        with embedding_duration.time():
            loop = asyncio.get_event_loop()
            # Run CPU-intensive embedding generation in thread pool
            embedding = await loop.run_in_executor(
                None, 
                self.embedding_model.encode,
                text,
                True  # normalize_embeddings
            )
            return embedding
    
    async def _vector_search(self,
                            embedding: np.ndarray,
                            collection_name: Optional[str],
                            n_results: int,
                            min_score: float) -> List[Dict[str, Any]]:
        """
        Execute vector similarity search using pgvector
        
        Args:
            embedding: Query embedding
            collection_name: Collection filter
            n_results: Number of results
            min_score: Minimum score threshold
            
        Returns:
            Search results
        """
        async with self.connection_pool.acquire() as conn:
            # Build query with pgvector operators
            query = """
                SELECT 
                    d.id as doc_id,
                    d.title,
                    d.content as text,
                    d.page,
                    d.metadata,
                    1 - (d.embedding <=> $1::vector) as score
                FROM documents d
                WHERE 
                    1 - (d.embedding <=> $1::vector) > $2
                    {collection_filter}
                ORDER BY d.embedding <=> $1::vector
                LIMIT $3
            """
            
            # Add collection filter if specified
            collection_filter = ""
            if collection_name:
                collection_filter = f"AND d.collection = '{collection_name}'"
            
            query = query.format(collection_filter=collection_filter)
            
            # Execute query
            rows = await conn.fetch(
                query,
                embedding.tolist(),
                min_score,
                n_results
            )
            
            # Format results
            results = []
            for row in rows:
                results.append({
                    'doc_id': row['doc_id'],
                    'title': row['title'],
                    'text': row['text'],
                    'page': row['page'],
                    'metadata': row['metadata'],
                    'score': float(row['score'])
                })
            
            return results
    
    async def _rerank_results(self, query: str, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rerank results using cross-encoder model
        
        Args:
            query: Original query
            results: Initial results
            
        Returns:
            Reranked results
        """
        if not results:
            return results
        
        try:
            # In production, would use actual reranking model
            # For now, just return top K based on original scores
            sorted_results = sorted(results, key=lambda x: x['score'], reverse=True)
            return sorted_results[:settings.RERANK_TOP_K]
            
        except Exception as e:
            logger.warning(f"Reranking failed, using original order: {str(e)}")
            return results
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            # Test database connection
            async with self.connection_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            
            # Test embedding model
            test_embedding = await self._generate_embedding_async("test")
            
            return len(test_embedding) == settings.EMBEDDING_DIMENSION
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise

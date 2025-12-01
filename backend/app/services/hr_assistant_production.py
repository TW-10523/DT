"""
Production-Grade HR Assistant Service
Includes async support, monitoring, caching, error handling, and security
"""

import asyncio
import json
import logging
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession
from prometheus_client import Counter, Histogram, Gauge
from opentelemetry import trace
from opentelemetry.instrumentation.logging import LoggingInstrumentor
import bleach
from tenacity import retry, stop_after_attempt, wait_exponential
from circuitbreaker import circuit

from app.core.config import settings
from app.services.hr_assistant_service import Source, Recommendation

# Configure structured logging
LoggingInstrumentor().instrument()
logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

# Metrics
query_counter = Counter('hr_assistant_queries_total', 'Total number of HR queries', ['status', 'language'])
query_duration = Histogram('hr_assistant_query_duration_seconds', 'Query processing duration')
cache_hits = Counter('hr_assistant_cache_hits_total', 'Cache hit count', ['cache_type'])
active_queries = Gauge('hr_assistant_active_queries', 'Number of active queries')
confidence_histogram = Histogram('hr_assistant_confidence_scores', 'Distribution of confidence scores')
translation_counter = Counter('hr_assistant_translations_total', 'Total translations', ['source_lang', 'target_lang'])
error_counter = Counter('hr_assistant_errors_total', 'Total errors', ['error_type'])

class ProductionHRAssistantService:
    """
    Production-grade HR Assistant with enterprise features
    """
    
    def __init__(self):
        """Initialize production HR assistant"""
        self.redis_client = None
        self.retriever_service = None
        self.translator_service = None
        self.llm_service = None
        self.initialized = False
        self._semaphore = asyncio.Semaphore(settings.ASYNC_MAX_CONNECTIONS)
        
    async def initialize(self):
        """Async initialization of services"""
        if self.initialized:
            return
        
        try:
            # Initialize Redis for caching
            if settings.CACHE_ENABLED:
                self.redis_client = await redis.Redis(
                    connection_pool=redis.ConnectionPool.from_url(
                        settings.REDIS_URL,
                        password=settings.REDIS_PASSWORD,
                        ssl=settings.REDIS_SSL,
                        max_connections=settings.REDIS_POOL_SIZE
                    )
                )
                await self.redis_client.ping()
                logger.info("Redis cache initialized")
            
            # Initialize services (would be actual service instances in production)
            from app.services.retriever_production import ProductionRetrieverService
            from app.services.translator_production import ProductionTranslatorService
            from app.services.llm_production import ProductionLLMService
            
            self.retriever_service = ProductionRetrieverService()
            self.translator_service = ProductionTranslatorService()
            self.llm_service = ProductionLLMService()

            if hasattr(self.retriever_service, "initialize"):
                await self.retriever_service.initialize()
            if hasattr(self.translator_service, "initialize"):
                # keep optional init for translator
                init = getattr(self.translator_service, "initialize")
                if asyncio.iscoroutinefunction(init):
                    await init()
            if hasattr(self.llm_service, "initialize"):
                await self.llm_service.initialize()
            
            self.initialized = True
            logger.info("HR Assistant Service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize HR Assistant: {str(e)}")
            raise
            
    
    async def close(self):
        """Cleanup resources"""
        try:
            if self.redis_client:
                await self.redis_client.close()
        except Exception:
            logger.exception("Error closing redis client")
        
        for svc in (self.retriever_service, self.translator_service, self.llm_service):
            if svc and hasattr(svc, "close"):
                try:
                    maybe = getattr(svc, "close")
                    if asyncio.iscoroutinefunction(maybe):
                        await maybe()
                    else:
                        maybe()
                except Exception:
                    logger.exception("Error closing service %s", svc)
        
        logger.info("HR Assistant Service closed")
    
    @asynccontextmanager
    async def _track_query(self):
        """Context manager for tracking active queries"""
        active_queries.inc()
        start_time = time.time()
        try:
            yield
        finally:
            active_queries.dec()
            duration = time.time() - start_time
            query_duration.observe(duration)
    
    @tracer.start_as_current_span("process_hr_query")
    async def process_query(self,
                           query: str,
                           collection_name: Optional[str] = None,
                           n_results: int = 5,
                           user_id: Optional[str] = None,
                           session: Optional[AsyncSession] = None) -> Dict[str, Any]:
        """
        Process HR query with production features
        
        Args:
            query: Sanitized user question
            collection_name: Optional collection to search
            n_results: Number of results to retrieve
            user_id: User identifier for tracking
            session: Async database session
            
        Returns:
            Formatted response with metadata
        """
        if not self.initialized:
            await self.initialize()
        
        async with self._track_query():
            async with self._semaphore:
                try:
                    # Input sanitization
                    query = self._sanitize_input(query)
                    
                    # Validate query length
                    if len(query) > settings.HR_MAX_QUERY_LENGTH:
                        raise ValueError(f"Query exceeds maximum length of {settings.HR_MAX_QUERY_LENGTH}")
                    
                    # Check cache first
                    cached_response = await self._check_cache(query, collection_name, n_results)
                    if cached_response:
                        cache_hits.labels(cache_type='full_response').inc()
                        logger.info(f"Cache hit for query: {query[:50]}...")
                        return cached_response
                    
                    # Process the query
                    response = await self._process_query_internal(
                        query, collection_name, n_results, user_id, session
                    )
                    
                    # Cache the response
                    await self._cache_response(query, collection_name, n_results, response)
                    
                    # Record metrics
                    query_counter.labels(status='success', language='auto').inc()
                    confidence_histogram.observe(response['metadata']['confidence'])
                    
                    # Log for analytics
                    try:
                        await self._log_query_analytics(query, response, user_id)
                    except Exception:
                        logger.exception("Failed logging analytics")
                    
                    return response
                    
                except Exception as e:
                    error_counter.labels(error_type=type(e).__name__).inc()
                    logger.error(f"Query processing error: {str(e)}", exc_info=True)
                    return self._create_error_response(str(e))
    
    def _sanitize_input(self, query: str) -> str:
        """
        Sanitize user input to prevent injection attacks
        
        Args:
            query: Raw user input
            
        Returns:
            Sanitized query
        """
        # Remove HTML tags and dangerous characters
        cleaned = bleach.clean(query, tags=[], strip=True)
        
        # Additional sanitization for SQL/NoSQL injection prevention
        cleaned = cleaned.replace("'", "''")
        cleaned = cleaned.replace('"', '""')
        cleaned = cleaned.replace(';', '')
        cleaned = cleaned.replace('--', '')
        
        return cleaned.strip()
    
    async def _check_cache(self, 
                          query: str, 
                          collection_name: Optional[str],
                          n_results: int) -> Optional[Dict[str, Any]]:
        """
        Check Redis cache for cached response
        
        Args:
            query: User query
            collection_name: Collection name
            n_results: Number of results
            
        Returns:
            Cached response or None
        """
        if not settings.CACHE_ENABLED or not self.redis_client:
            return None
        
        try:
            cache_key = self._generate_cache_key(query, collection_name, n_results)
            cached = await self.redis_client.get(cache_key)
            
            if cached:
                return json.loads(cached)
            
        except Exception as e:
            logger.warning(f"Cache lookup failed: {str(e)}")
        
        return None
    
    def _generate_cache_key(self, 
                           query: str, 
                           collection_name: Optional[str],
                           n_results: int) -> str:
        """Generate cache key for query"""
        key_parts = [
            query.lower(),
            collection_name or "default",
            str(n_results)
        ]
        key_string = "|".join(key_parts)
        return f"hr_query:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _cache_response(self,
                             query: str,
                             collection_name: Optional[str],
                             n_results: int,
                             response: Dict[str, Any]) -> None:
        """Cache response in Redis"""
        if not settings.CACHE_ENABLED or not self.redis_client:
            return
        
        try:
            cache_key = self._generate_cache_key(query, collection_name, n_results)
            await self.redis_client.setex(
                cache_key,
                settings.CACHE_TTL_SECONDS,
                json.dumps(response)
            )
        except Exception as e:
            logger.warning(f"Failed to cache response: {str(e)}")
    
    @retry(
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.LLM_RETRY_DELAY)
    )
    @circuit(
        failure_threshold=settings.CIRCUIT_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=settings.CIRCUIT_BREAKER_RECOVERY_TIMEOUT
    )
    async def _process_query_internal(self,
                                     query: str,
                                     collection_name: Optional[str],
                                     n_results: int,
                                     user_id: Optional[str],
                                     session: Optional[AsyncSession]) -> Dict[str, Any]:
        """
        Internal query processing with retry and circuit breaker
        
        Args:
            query: Sanitized query
            collection_name: Collection name
            n_results: Number of results
            user_id: User ID
            session: Database session
            
        Returns:
            Processed response
        """
        # Step 1: Retrieve passages (with timeout)
        try:
            passages = await asyncio.wait_for(
                self._retrieve_passages_async(query, collection_name, n_results, session),
                timeout=settings.SEARCH_TIMEOUT_SECONDS
            )
        except asyncio.TimeoutError:
            logger.error("Passage retrieval timeout")
            return self._create_timeout_response()
        
        if not passages:
            return self._create_no_results_response()
        
        # Step 2: Process and translate passages
        processed_sources = await self._process_passages_async(passages)
        
        # Step 3: Generate answer with LLM
        answer_lines, sources, recommendations, confidence = await self._generate_answer_async(
            query, processed_sources
        )
        
        # Step 4: Format response
        return self._format_response(answer_lines, sources, recommendations, confidence)
    
    async def _retrieve_passages_async(self,
                                      query: str,
                                      collection_name: Optional[str],
                                      n_results: int,
                                      session: Optional[AsyncSession]) -> List[Dict[str, Any]]:
        """
        Async retrieval of passages from vector database
        
        Args:
            query: Search query
            collection_name: Collection to search
            n_results: Number of results
            session: Database session
            
        Returns:
            Retrieved passages
        """
        with tracer.start_as_current_span("retrieve_passages"):
            try:
                results = await self.retriever_service.search_async(
                    query=query,
                    collection_name=collection_name,
                    n_results=n_results,
                    min_score=settings.SEARCH_MIN_SCORE,
                    session=session
                )
                
                logger.info(f"Retrieved {len(results)} passages for query")
                return results
                
            except Exception as e:
                logger.error(f"Retrieval error: {str(e)}")
                raise
    
    async def _process_passages_async(self, passages: List[Dict[str, Any]]) -> List[Source]:
        """
        Async processing of passages with translation
        
        Args:
            passages: Raw passages
            
        Returns:
            Processed Source objects
        """
        with tracer.start_as_current_span("process_passages"):
            # Process passages in parallel
            tasks = [self._process_single_passage(p) for p in passages]
            processed = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out any failures
            sources = []
            for result in processed:
                if isinstance(result, Source):
                    sources.append(result)
                elif isinstance(result, Exception):
                    logger.warning(f"Failed to process passage: {str(result)}")
            
            return sources
    
    async def _process_single_passage(self, passage: Dict[str, Any]) -> Source:
        """Process a single passage with translation"""
        text = passage.get('text', '')
        
        # Detect language
        lang = await self.translator_service.detect_language_async(text)
        
        # Translate if needed
        if lang != 'en' and settings.FEATURE_MULTILINGUAL:
            translated = await self.translator_service.translate_async(
                text, source_lang=lang, target_lang='en'
            )
            translation_counter.labels(source_lang=lang, target_lang='en').inc()
        else:
            translated = text
        
        return Source(
            doc_id=str(passage.get('doc_id', '')),
            title=passage.get('title', 'Untitled'),
            page=passage.get('page', 0),
            original_snippet=text,
            translated_snippet=translated,
            score=float(passage.get('score', 0.0))
        )
    
    async def _generate_answer_async(self,
                                    query: str,
                                    sources: List[Source]) -> Tuple[List[str], List[Source], List[Recommendation], float]:
        """
        Generate answer using async LLM service
        
        Args:
            query: User query
            sources: Processed sources
            
        Returns:
            Answer components
        """
        with tracer.start_as_current_span("generate_answer"):
            # Prepare context
            context = self._prepare_context(sources)
            
            # Create prompt
            prompt = self._create_production_prompt(query, context)
            
            # Generate with timeout
            try:
                llm_response = await asyncio.wait_for(
                    self.llm_service.generate_async(
                        prompt=prompt,
                        temperature=settings.LLM_TEMPERATURE,
                        max_tokens=settings.LLM_MAX_TOKENS
                    ),
                    timeout=settings.LLM_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                logger.error("LLM generation timeout")
                return self._create_fallback_answer(query, sources)
            
            # Parse response
            return self._parse_llm_response(llm_response, sources)
    
    def _prepare_context(self, sources: List[Source]) -> str:
        """Prepare context from sources"""
        context_parts = []
        for i, source in enumerate(sources[:settings.SEARCH_MAX_RESULTS], 1):
            context_parts.append(
                f"[Source {i}] (Doc: {source.doc_id}, Page: {source.page}, Score: {source.score:.2f})\n"
                f"{source.translated_snippet[:500]}\n"  # Limit snippet length
            )
        return "\n".join(context_parts)
    
    def _create_production_prompt(self, query: str, context: str) -> str:
        """Create production-optimized prompt"""
        return f"""You are an expert HR assistant providing accurate, evidence-based answers.

STRICT RULES:
1. Use ONLY information from the provided sources
2. Generate EXACTLY 4 lines of meaningful English text
3. If sources conflict, mention this and reduce confidence
4. Never hallucinate or add information not in sources
5. If no relevant information, state clearly

SOURCES:
{context}

QUERY: {query}

Generate exactly 4 lines of answer text (no more, no less):"""
    
    def _parse_llm_response(self,
                           llm_text: str,
                           sources: List[Source]) -> Tuple[List[str], List[Source], List[Recommendation], float]:
        """Parse and validate LLM response"""
        lines = llm_text.strip().split('\n')
        
        # Ensure exactly 4 lines
        if len(lines) < 4:
            lines.extend(['Additional information not available.'] * (4 - len(lines)))
        answer_lines = lines[:4]
        
        # Validate each line
        for i, line in enumerate(answer_lines):
            if not line or len(line.strip()) < 2:
                answer_lines[i] = "Additional context unavailable for this aspect."
        
        # Calculate confidence
        confidence = self._calculate_production_confidence(sources, answer_lines)
        
        # Generate recommendations
        recommendations = self._generate_smart_recommendations(answer_lines[0], sources)
        
        # Filter to top sources
        used_sources = sorted(sources, key=lambda s: s.score, reverse=True)[:3]
        
        return answer_lines, used_sources, recommendations, confidence
    
    def _calculate_production_confidence(self, sources: List[Source], answer_lines: List[str]) -> float:
        """Calculate confidence with production logic"""
        if not sources:
            return 0.0
        
        # Base confidence on source quality
        avg_score = sum(s.score for s in sources[:3]) / min(3, len(sources))
        
        # Adjust for answer quality indicators
        answer_text = ' '.join(answer_lines).lower()
        
        # Reduce for conflicts
        if any(word in answer_text for word in ['conflict', 'disagree', 'varies', 'different']):
            avg_score *= settings.HR_CONFLICT_CONFIDENCE_MULTIPLIER
        
        # Reduce for no information
        if 'no authoritative' in answer_text or 'not found' in answer_text:
            return 0.1
        
        # Apply minimum threshold
        return max(min(avg_score, 1.0), settings.HR_MIN_CONFIDENCE_THRESHOLD)
    
    def _generate_smart_recommendations(self, 
                                       first_line: str, 
                                       sources: List[Source]) -> List[Recommendation]:
        """Generate contextually relevant recommendations"""
        recommendations = []
        query_lower = first_line.lower()
        
        # Smart recommendation logic based on topic
        topic_recommendations = {
            'leave': [
                Recommendation("Check leave balance", "View your current accruals"),
                Recommendation("Request time off", "Submit leave request online")
            ],
            'benefit': [
                Recommendation("Compare plans", "View plan options and costs"),
                Recommendation("Enrollment guide", "Step-by-step enrollment help")
            ],
            'policy': [
                Recommendation("Policy library", "Browse all HR policies"),
                Recommendation("Recent updates", "View latest policy changes")
            ],
            'salary': [
                Recommendation("Compensation review", "Understand salary structure"),
                Recommendation("Performance metrics", "View performance criteria")
            ],
            'training': [
                Recommendation("Learning catalog", "Browse available courses"),
                Recommendation("Skill assessments", "Identify development areas")
            ]
        }
        
        # Find matching topics
        for topic, recs in topic_recommendations.items():
            if topic in query_lower:
                recommendations.extend(recs)
        
        # Add general recommendation if none found
        if not recommendations:
            recommendations.append(
                Recommendation("Contact HR", "Get personalized assistance")
            )
        
        return recommendations[:settings.HR_MAX_RECOMMENDATIONS]
    
    def _format_response(self,
                        answer_lines: List[str],
                        sources: List[Source],
                        recommendations: List[Recommendation],
                        confidence: float) -> Dict[str, Any]:
        """Format production response"""
        # Create metadata
        metadata = {
            "sources": [s.to_dict() for s in sources],
            "recommendations": [r.to_dict() for r in recommendations],
            "confidence": round(confidence, 2)
        }
        
        # Create formatted output
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION
        }
    
    def _create_fallback_answer(self, 
                               query: str, 
                               sources: List[Source]) -> Tuple[List[str], List[Source], List[Recommendation], float]:
        """Create fallback answer when LLM fails"""
        answer_lines = [
            "Unable to generate a complete answer at this time.",
            f"Found {len(sources)} relevant documents for your query.",
            "Please review the source documents or try rephrasing your question.",
            "You may also contact HR directly for immediate assistance."
        ]
        
        recommendations = [
            Recommendation("Try again", "Rephrase your question"),
            Recommendation("Contact HR", "Get direct assistance")
        ]
        
        return answer_lines, sources[:3], recommendations, 0.3
    
    def _create_no_results_response(self) -> Dict[str, Any]:
        """Create response when no results found"""
        answer_lines = [
            "No authoritative answer found in the docs.",
            "The query did not match any relevant HR documentation.",
            "Please try rephrasing your question or contact HR directly.",
            "You may also browse the HR portal for general information."
        ]
        
        metadata = {
            "sources": [],
            "recommendations": [
                {"title": "Browse HR Portal", "reason": "Access all HR resources"},
                {"title": "Contact HR Team", "reason": "Get direct assistance"},
                {"title": "FAQ Section", "reason": "Common questions and answers"}
            ],
            "confidence": 0.0
        }
        
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION
        }
    
    def _create_timeout_response(self) -> Dict[str, Any]:
        """Create response for timeout scenarios"""
        answer_lines = [
            "The system is experiencing high load.",
            "Your request could not be completed in time.",
            "Please try again in a few moments.",
            "If the issue persists, contact IT support."
        ]
        
        metadata = {
            "sources": [],
            "recommendations": [
                {"title": "Retry", "reason": "Try your request again"},
                {"title": "Contact Support", "reason": "Report technical issues"}
            ],
            "confidence": 0.0
        }
        
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata,
            "error_type": "timeout",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION
        }
    
    def _create_error_response(self, error_msg: str) -> Dict[str, Any]:
        """Create response for error conditions"""
        # Don't expose internal errors to users
        safe_error = "An error occurred processing your request."
        if "query exceeds maximum length" in error_msg.lower():
            safe_error = "Your question is too long. Please shorten it and try again."
        
        answer_lines = [
            safe_error,
            "The system encountered an issue with your request.",
            "Please try again or contact support if the issue persists.",
            "Your query has been logged for review."
        ]
        
        metadata = {
            "sources": [],
            "recommendations": [
                {"title": "Retry Query", "reason": "Try submitting again"},
                {"title": "Simplify Question", "reason": "Use simpler phrasing"},
                {"title": "Contact Support", "reason": "Get technical help"}
            ],
            "confidence": 0.0
        }
        
        formatted_output = f"{chr(10).join(answer_lines)}\n{json.dumps(metadata, separators=(',', ':'))}"
        
        # Log actual error for debugging
        logger.error(f"Error response generated: {error_msg}")
        
        return {
            "formatted_response": formatted_output,
            "answer_lines": answer_lines,
            "metadata": metadata,
            "error": safe_error,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION
        }
    
    async def _log_query_analytics(self, 
                                  query: str, 
                                  response: Dict[str, Any],
                                  user_id: Optional[str]) -> None:
        """Log query for analytics and improvement"""
        if not settings.FEATURE_ANALYTICS:
            return
        
        try:
            analytics_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "user_id": user_id or "anonymous",
                "query": query[:200],  # Truncate for storage
                "confidence": response['metadata']['confidence'],
                "source_count": len(response['metadata']['sources']),
                "has_results": len(response['metadata']['sources']) > 0,
                "response_time_ms": int(query_duration._sum.get()),
            }
            
            # In production, would send to analytics service
            logger.info(f"Query analytics: {json.dumps(analytics_data)}")
            
        except Exception as e:
            logger.warning(f"Failed to log analytics: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for monitoring"""
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "checks": {}
        }
        
        # Check Redis
        if settings.CACHE_ENABLED:
            try:
                await self.redis_client.ping()
                health_status["checks"]["redis"] = "healthy"
            except Exception as e:
                health_status["checks"]["redis"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        # Check retriever service
        if self.retriever_service:
            try:
                await self.retriever_service.health_check()
                health_status["checks"]["retriever"] = "healthy"
            except Exception as e:
                health_status["checks"]["retriever"] = f"unhealthy: {str(e)}"
                health_status["status"] = "unhealthy"
        
        # Check LLM service
        if self.llm_service:
            try:
                await self.llm_service.health_check()
                health_status["checks"]["llm"] = "healthy"
            except Exception as e:
                health_status["checks"]["llm"] = f"unhealthy: {str(e)}"
                health_status["status"] = "degraded"
        
        return health_status

# Global instance with lazy initialization
_hr_assistant_instance = None

async def get_hr_assistant() -> ProductionHRAssistantService:
    """Get or create singleton HR assistant instance"""
    global _hr_assistant_instance
    
    if _hr_assistant_instance is None:
        _hr_assistant_instance = ProductionHRAssistantService()
        await _hr_assistant_instance.initialize()
    
    return _hr_assistant_instance

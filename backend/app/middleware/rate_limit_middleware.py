
"""
Rate Limiting Middleware
Implements token bucket algorithm with Redis backend
"""

import logging
import time
import hashlib
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta

import redis.asyncio as redis
from fastapi import Request, HTTPException, status
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
rate_limit_hits = Counter('rate_limit_hits_total', 'Rate limit hits', ['limit_type'])
rate_limit_allowed = Counter('rate_limit_allowed_total', 'Requests allowed by rate limiter')
rate_limit_denied = Counter('rate_limit_denied_total', 'Requests denied by rate limiter')

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware using token bucket algorithm
    """
    
    def __init__(self, app, redis_client: Optional[redis.Redis] = None):
        super().__init__(app)
        self.redis_client = redis_client
        self.local_buckets: Dict[str, Dict] = {}  # Fallback for when Redis is unavailable
        
    async def dispatch(self, request: Request, call_next):
        """Apply rate limiting to requests"""
        
        # Skip rate limiting if disabled
        if not settings.RATE_LIMIT_ENABLED:
            return await call_next(request)
        
        # Skip for health check endpoints
        if request.url.path in [
            settings.HEALTH_CHECK_PATH,
            settings.READINESS_CHECK_PATH,
            settings.LIVENESS_CHECK_PATH,
            settings.METRICS_PATH
        ]:
            return await call_next(request)
        
        # Get client identifier
        client_id = self._get_client_id(request)
        
        # Check rate limits
        try:
            allowed, retry_after = await self._check_rate_limit(client_id, request)
            
            if not allowed:
                rate_limit_denied.inc()
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={
                        "Retry-After": str(retry_after),
                        "X-RateLimit-Limit": str(settings.RATE_LIMIT_PER_MINUTE),
                        "X-RateLimit-Remaining": "0",
                        "X-RateLimit-Reset": str(int(time.time()) + retry_after)
                    }
                )
            
            rate_limit_allowed.inc()
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Rate limiting error: {str(e)}")
            # Allow request on error to avoid blocking legitimate traffic
            rate_limit_allowed.inc()
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        try:
            remaining = await self._get_remaining_requests(client_id)
            response.headers["X-RateLimit-Limit"] = str(settings.RATE_LIMIT_PER_MINUTE)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time()) + 60)
        except:
            pass  # Don't fail request if we can't add headers
        
        return response
    
    def _get_client_id(self, request: Request) -> str:
        """
        Get unique client identifier from request
        
        Args:
            request: FastAPI request
            
        Returns:
            Client identifier string
        """
        # Priority: API key > JWT user ID > IP address
        
        # Check for API key
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if api_key:
            return f"api_key:{hashlib.md5(api_key.encode()).hexdigest()}"
        
        # Check for authenticated user
        if hasattr(request.state, 'user'):
            user = request.state.user
            if user.get('user_id'):
                return f"user:{user['user_id']}"
        
        # Fall back to IP address
        client_ip = request.client.host if request.client else "unknown"
        
        # Check for X-Forwarded-For header (when behind proxy)
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    async def _check_rate_limit(self, 
                               client_id: str, 
                               request: Request) -> Tuple[bool, int]:
        """
        Check if request is within rate limits
        
        Args:
            client_id: Client identifier
            request: FastAPI request
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        # Use Redis if available
        if self.redis_client:
            return await self._check_redis_rate_limit(client_id)
        else:
            return self._check_local_rate_limit(client_id)
    
    async def _check_redis_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Check rate limit using Redis
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Use multiple time windows
        limits = [
            (f"{client_id}:minute", settings.RATE_LIMIT_PER_MINUTE, 60),
            (f"{client_id}:hour", settings.RATE_LIMIT_PER_HOUR, 3600),
            (f"{client_id}:day", settings.RATE_LIMIT_PER_DAY, 86400)
        ]
        
        for key, limit, window in limits:
            # Use Redis sorted set to track requests
            pipe = self.redis_client.pipeline()
            
            # Remove old entries
            pipe.zremrangebyscore(key, 0, now - window)
            
            # Count recent requests
            pipe.zcard(key)
            
            # Add current request
            pipe.zadd(key, {str(now): now})
            
            # Set expiry
            pipe.expire(key, window)
            
            results = await pipe.execute()
            count = results[1]  # Get count before adding current request
            
            if count >= limit:
                # Rate limit exceeded
                rate_limit_hits.labels(limit_type=key.split(':')[-1]).inc()
                
                # Calculate retry after
                oldest_score = await self.redis_client.zrange(key, 0, 0, withscores=True)
                if oldest_score:
                    oldest_time = oldest_score[0][1]
                    retry_after = int(window - (now - oldest_time)) + 1
                else:
                    retry_after = window
                
                return False, retry_after
        
        return True, 0
    
    def _check_local_rate_limit(self, client_id: str) -> Tuple[bool, int]:
        """
        Check rate limit using local token bucket (fallback)
        
        Args:
            client_id: Client identifier
            
        Returns:
            Tuple of (allowed, retry_after_seconds)
        """
        now = time.time()
        
        # Get or create bucket
        if client_id not in self.local_buckets:
            self.local_buckets[client_id] = {
                'tokens': settings.RATE_LIMIT_PER_MINUTE,
                'last_update': now,
                'minute_requests': [],
                'hour_requests': [],
                'day_requests': []
            }
        
        bucket = self.local_buckets[client_id]
        
        # Clean old requests
        bucket['minute_requests'] = [
            t for t in bucket['minute_requests'] if now - t < 60
        ]
        bucket['hour_requests'] = [
            t for t in bucket['hour_requests'] if now - t < 3600
        ]
        bucket['day_requests'] = [
            t for t in bucket['day_requests'] if now - t < 86400
        ]
        
        # Check limits
        if len(bucket['minute_requests']) >= settings.RATE_LIMIT_PER_MINUTE:
            rate_limit_hits.labels(limit_type='minute').inc()
            retry_after = 60 - (now - bucket['minute_requests'][0])
            return False, int(retry_after) + 1
        
        if len(bucket['hour_requests']) >= settings.RATE_LIMIT_PER_HOUR:
            rate_limit_hits.labels(limit_type='hour').inc()
            retry_after = 3600 - (now - bucket['hour_requests'][0])
            return False, int(retry_after) + 1
        
        if len(bucket['day_requests']) >= settings.RATE_LIMIT_PER_DAY:
            rate_limit_hits.labels(limit_type='day').inc()
            retry_after = 86400 - (now - bucket['day_requests'][0])
            return False, int(retry_after) + 1
        
        # Add request
        bucket['minute_requests'].append(now)
        bucket['hour_requests'].append(now)
        bucket['day_requests'].append(now)
        
        # Clean up old clients periodically
        if len(self.local_buckets) > 1000:
            # Remove clients that haven't made requests in the last hour
            cutoff = now - 3600
            to_remove = [
                cid for cid, b in self.local_buckets.items()
                if b['last_update'] < cutoff
            ]
            for cid in to_remove:
                del self.local_buckets[cid]
        
        bucket['last_update'] = now
        
        return True, 0
    
    async def _get_remaining_requests(self, client_id: str) -> int:
        """
        Get remaining requests for client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Number of remaining requests
        """
        if self.redis_client:
            key = f"{client_id}:minute"
            count = await self.redis_client.zcard(key)
            return max(0, settings.RATE_LIMIT_PER_MINUTE - count)
        else:
            bucket = self.local_buckets.get(client_id, {})
            minute_requests = bucket.get('minute_requests', [])
            now = time.time()
            recent = [t for t in minute_requests if now - t < 60]
            return max(0, settings.RATE_LIMIT_PER_MINUTE - len(recent))

class IPWhitelistMiddleware(BaseHTTPMiddleware):
    """
    IP whitelist/blacklist middleware
    """
    
    def __init__(self, app, whitelist: Optional[list[str]] = None, blacklist: Optional[list[str]] = None):
        super().__init__(app)
        self.whitelist = set(whitelist) if whitelist else None
        self.blacklist = set(blacklist) if blacklist else set()
        
    async def dispatch(self, request: Request, call_next):
        """Check IP against whitelist/blacklist"""
        
        # Get client IP
        client_ip = request.client.host if request.client else None
        
        # Check for X-Forwarded-For header
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            client_ip = forwarded.split(",")[0].strip()
        
        if not client_ip:
            # Can't determine IP, allow for now
            return await call_next(request)
        
        # Check blacklist
        if client_ip in self.blacklist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        # Check whitelist (if configured)
        if self.whitelist and client_ip not in self.whitelist:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied"
            )
        
        return await call_next(request)

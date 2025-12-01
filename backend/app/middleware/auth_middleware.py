"""
Authentication and Authorization Middleware
Provides API key validation, JWT support, and role-based access
"""

import logging
import time
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

import jwt
from fastapi import Request, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_client import Counter

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['method', 'status'])
auth_failures = Counter('auth_failures_total', 'Total authentication failures', ['reason'])

# Security schemes
api_key_header = APIKeyHeader(name=settings.API_KEY_HEADER, auto_error=False)
bearer_scheme = HTTPBearer(auto_error=False)

class AuthMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for production API
    """
    
    async def dispatch(self, request: Request, call_next):
        """Process authentication for each request"""
        
        # Skip auth for health check endpoints
        if request.url.path in [
            settings.HEALTH_CHECK_PATH,
            settings.READINESS_CHECK_PATH,
            settings.LIVENESS_CHECK_PATH,
            settings.METRICS_PATH
        ]:
            return await call_next(request)
        
        # Skip auth for docs in non-production
        if settings.ENVIRONMENT != "production" and request.url.path in ["/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)
        
        # Authenticate request
        try:
            auth_result = await self.authenticate_request(request)
            
            if not auth_result['authenticated']:
                auth_failures.labels(reason=auth_result.get('reason', 'unknown')).inc()
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail=auth_result.get('message', 'Authentication required'),
                    headers={"WWW-Authenticate": "Bearer"}
                )
            
            # Add user info to request state
            request.state.user = auth_result.get('user', {})
            request.state.auth_method = auth_result.get('method', 'unknown')
            
            auth_attempts.labels(method=auth_result['method'], status='success').inc()
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            auth_failures.labels(reason='error').inc()
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication service error"
            )
        
        # Process request
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        return response
    
    async def authenticate_request(self, request: Request) -> Dict[str, Any]:
        """
        Authenticate request using API key or JWT
        
        Args:
            request: FastAPI request
            
        Returns:
            Authentication result dictionary
        """
        # Try API key authentication first
        api_key = request.headers.get(settings.API_KEY_HEADER)
        if api_key:
            return await self.validate_api_key(api_key)
        
        # Try JWT authentication
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.split(" ")[1]
            return await self.validate_jwt(token)
        
        # No authentication provided
        return {
            'authenticated': False,
            'reason': 'no_auth',
            'message': 'No authentication credentials provided'
        }
    
    async def validate_api_key(self, api_key: str) -> Dict[str, Any]:
        """
        Validate API key
        
        Args:
            api_key: API key to validate
            
        Returns:
            Validation result
        """
        if not settings.API_KEYS:
            # API keys not configured
            return {
                'authenticated': False,
                'reason': 'not_configured',
                'message': 'API key authentication not configured'
            }
        
        if api_key in settings.API_KEYS:
            return {
                'authenticated': True,
                'method': 'api_key',
                'user': {
                    'type': 'api_key',
                    'key_id': api_key[:8] + '...'  # Partial key for logging
                }
            }
        
        return {
            'authenticated': False,
            'reason': 'invalid_key',
            'message': 'Invalid API key'
        }
    
    async def validate_jwt(self, token: str) -> Dict[str, Any]:
        """
        Validate JWT token
        
        Args:
            token: JWT token
            
        Returns:
            Validation result
        """
        try:
            # Decode and verify JWT
            payload = jwt.decode(
                token,
                settings.JWT_SECRET,
                algorithms=[settings.JWT_ALGORITHM]
            )
            
            # Check expiration
            exp = payload.get('exp')
            if exp and datetime.fromtimestamp(exp) < datetime.utcnow():
                return {
                    'authenticated': False,
                    'reason': 'token_expired',
                    'message': 'Token has expired'
                }
            
            return {
                'authenticated': True,
                'method': 'jwt',
                'user': {
                    'type': 'jwt',
                    'user_id': payload.get('sub'),
                    'email': payload.get('email')
                }
            }
            
        except jwt.InvalidTokenError as e:
            return {
                'authenticated': False,
                'reason': 'invalid_token',
                'message': f'Invalid token: {str(e)}'
            }
        except Exception as e:
            logger.error(f"JWT validation error: {str(e)}")
            return {
                'authenticated': False,
                'reason': 'validation_error',
                'message': 'Token validation failed'
            }

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token
    
    Args:
        data: Token payload
        expires_delta: Token expiration time
        
    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=settings.JWT_EXPIRATION_HOURS)
    
    to_encode.update({"exp": expire})
    
    encoded_jwt = jwt.encode(
        to_encode,
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM
    )
    
    return encoded_jwt

async def get_current_user(request: Request) -> Dict[str, Any]:
    """
    Get current authenticated user from request
    
    Args:
        request: FastAPI request
        
    Returns:
        User information dictionary
    """
    if not hasattr(request.state, 'user'):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated"
        )
    
    return request.state.user


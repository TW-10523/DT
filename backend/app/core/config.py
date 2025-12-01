"""
Production Configuration for HR Assistant
Includes settings for database, caching, security, monitoring, and external services
"""

from pydantic_settings import BaseSettings
from typing import Optional, List
import secrets
from enum import Enum

class Environment(str, Enum):
    """Application environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class settings(BaseSettings):
    """
    Production-grade settings for HR Assistant
    All sensitive values should be loaded from environment variables
    """
    
    # Environment
    ENVIRONMENT: Environment = Environment.PRODUCTION
    DEBUG: bool = False
    
    # Application
    APP_NAME: str = "HR Assistant API"
    APP_VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Security
    SECRET_KEY: str = secrets.token_urlsafe(32)
    API_KEY_HEADER: str = "X-API-Key"
    API_KEYS: List[str] = []  # List of valid API keys
    JWT_SECRET: str = secrets.token_urlsafe(32)
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24
    ALLOWED_ORIGINS: List[str] = ["*"]
    ENABLE_CORS: bool = True
    
    # Rate Limiting
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_PER_MINUTE: int = 60
    RATE_LIMIT_PER_HOUR: int = 1000
    RATE_LIMIT_PER_DAY: int = 10000
    
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:password@localhost/hrdb"
    DATABASE_POOL_SIZE: int = 20
    DATABASE_MAX_OVERFLOW: int = 40
    DATABASE_POOL_TIMEOUT: int = 30
    DATABASE_POOL_RECYCLE: int = 3600
    DATABASE_ECHO: bool = False
    
    # Vector Database
    VECTOR_DB_URL: str = "postgresql://user:password@localhost/vectordb"
    VECTOR_DIMENSION: int = 1024
    VECTOR_INDEX_TYPE: str = "ivfflat"
    VECTOR_INDEX_LISTS: int = 100
    
    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_PASSWORD: Optional[str] = None
    REDIS_SSL: bool = False
    REDIS_POOL_SIZE: int = 10
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True
    
    # LLM Service
    LLM_ENDPOINT: str = "http://localhost:8080"
    LLM_API_KEY: Optional[str] = None
    LLM_MODEL_NAME: str = "gpt-4"
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_TOKENS: int = 500
    LLM_TIMEOUT_SECONDS: int = 30
    LLM_MAX_RETRIES: int = 3
    LLM_RETRY_DELAY: float = 1.0
    
    # Translation Service
    TRANSLATION_PROVIDER: str = "deepl"  # deepl, google, azure, custom
    TRANSLATION_API_KEY: Optional[str] = None
    TRANSLATION_ENDPOINT: Optional[str] = None
    TRANSLATION_CACHE_ENABLED: bool = True
    TRANSLATION_MAX_CACHE_SIZE: int = 10000
    TRANSLATION_TIMEOUT_SECONDS: int = 10
    
    # Embedding Service
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024
    EMBEDDING_BATCH_SIZE: int = 32
    EMBEDDING_DEVICE: str = "cuda"  # cuda, cpu, mps
    EMBEDDING_CACHE_DIR: str = "./models/embeddings"
    
    # Reranking
    RERANK_MODEL_NAME: str = "hotchpotch/japanese-bge-reranker-v2-m3-v1"
    RERANK_MODEL_CACHE: str = "./models/rerank"
    RERANK_TOP_K: int = 10
    RERANK_ENABLED: bool = True
    
    # Search Configuration
    SEARCH_MIN_SCORE: float = 0.5
    SEARCH_MAX_RESULTS: int = 20
    SEARCH_DEFAULT_RESULTS: int = 5
    SEARCH_TIMEOUT_SECONDS: int = 10
    
    # HR Assistant Specific
    HR_MAX_QUERY_LENGTH: int = 500
    HR_MIN_CONFIDENCE_THRESHOLD: float = 0.3
    HR_CONFLICT_CONFIDENCE_MULTIPLIER: float = 0.7
    HR_MAX_RECOMMENDATIONS: int = 3
    HR_ANSWER_LINES: int = 4
    HR_DEFAULT_LANGUAGE: str = "en"
    HR_SUPPORTED_LANGUAGES: List[str] = ["en", "ja", "zh", "ko", "es", "fr", "de"]
    
    # Logging
    LOG_LEVEL: LogLevel = LogLevel.INFO
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[str] = "hr_assistant.log"
    LOG_MAX_BYTES: int = 10485760  # 10MB
    LOG_BACKUP_COUNT: int = 5
    LOG_JSON_FORMAT: bool = True
    
    # Monitoring
    METRICS_ENABLED: bool = True
    METRICS_PORT: int = 9090
    METRICS_PATH: str = "/metrics"
    TRACING_ENABLED: bool = True
    TRACING_ENDPOINT: Optional[str] = "http://localhost:4317"
    TRACING_SERVICE_NAME: str = "hr-assistant"
    
    # Health Checks
    HEALTH_CHECK_PATH: str = "/health"
    READINESS_CHECK_PATH: str = "/ready"
    LIVENESS_CHECK_PATH: str = "/alive"
    
    # Circuit Breaker
    CIRCUIT_BREAKER_ENABLED: bool = True
    CIRCUIT_BREAKER_FAILURE_THRESHOLD: int = 5
    CIRCUIT_BREAKER_RECOVERY_TIMEOUT: int = 60
    CIRCUIT_BREAKER_EXPECTED_EXCEPTION: str = "Exception"
    
    # Async Configuration
    ASYNC_ENABLED: bool = True
    ASYNC_WORKER_THREADS: int = 4
    ASYNC_MAX_CONNECTIONS: int = 100
    
    # File Upload
    MAX_UPLOAD_SIZE_MB: int = 10
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".pdf", ".txt", ".docx", ".md"]
    UPLOAD_TEMP_DIR: str = "/tmp/hr_uploads"
    
    # Data Retention
    RETENTION_DAYS: int = 90
    CLEANUP_ENABLED: bool = True
    CLEANUP_CRON: str = "0 2 * * *"  # 2 AM daily
    
    # Feature Flags
    FEATURE_MULTILINGUAL: bool = True
    FEATURE_FEEDBACK: bool = True
    FEATURE_DOCUMENT_UPLOAD: bool = True
    FEATURE_ANALYTICS: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
        # Allow extra fields for forward compatibility
        extra = "allow"

# Singleton instance
settings = settings()

# Validation
def validate_settings():
    """Validate critical settings on startup"""
    errors = []
    
    # Check database URLs
    if not settings.DATABASE_URL:
        errors.append("DATABASE_URL is required")
    
    if not settings.VECTOR_DB_URL:
        errors.append("VECTOR_DB_URL is required")
    
    # Check API keys in production
    if settings.ENVIRONMENT == Environment.PRODUCTION:
        if not settings.API_KEYS:
            errors.append("API_KEYS must be set in production")
        
        if settings.DEBUG:
            errors.append("DEBUG must be False in production")
        
        if not settings.LLM_API_KEY:
            errors.append("LLM_API_KEY is required in production")
        
        if settings.TRANSLATION_PROVIDER != "custom" and not settings.TRANSLATION_API_KEY:
            errors.append("TRANSLATION_API_KEY is required for external providers")
    
    # Check Redis if caching is enabled
    if settings.CACHE_ENABLED and not settings.REDIS_URL:
        errors.append("REDIS_URL is required when caching is enabled")
    
    # Check monitoring endpoints
    if settings.TRACING_ENABLED and not settings.TRACING_ENDPOINT:
        errors.append("TRACING_ENDPOINT is required when tracing is enabled")
    
    if errors:
        raise ValueError(f"Configuration errors: {', '.join(errors)}")
    
    return True

# Export settings
__all__ = ["settings", "validate_settings", "Environment", "LogLevel"]

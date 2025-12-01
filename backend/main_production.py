"""
Production FastAPI Application
Full-featured production application with all middleware and monitoring
"""

import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
import redis.asyncio as redis
import structlog
from app.core.config import settings, validate_settings
from app.api.hr_assistant_api import router as router
from app.api.search_api import router as search_router
from app.middleware.auth_middleware import AuthMiddleware
from app.middleware.rate_limit_middleware import RateLimitMiddleware
from app.services.hr_assistant_production import get_hr_assistant

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() if settings.LOG_JSON_FORMAT else structlog.dev.ConsoleRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Global Redis client
redis_client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown
    """
    global redis_client
    
    # Startup
    try:
        logger.info("Starting HR Assistant Application", 
                   version=settings.APP_VERSION,
                   environment=settings.ENVIRONMENT)
        
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
        
        # Initialize HR Assistant service
        hr_assistant = await get_hr_assistant()
        logger.info("HR Assistant service initialized")
        
        # Initialize OpenTelemetry tracing
        if settings.TRACING_ENABLED:
            resource = Resource.create({
                "service.name": settings.TRACING_SERVICE_NAME,
                "service.version": settings.APP_VERSION,
                "deployment.environment": settings.ENVIRONMENT
            })
            
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            otlp_exporter = OTLPSpanExporter(
                endpoint=settings.TRACING_ENDPOINT,
                insecure=True
            )
            
            span_processor = BatchSpanProcessor(otlp_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            logger.info("OpenTelemetry tracing initialized")
        
        logger.info("Application startup complete")
        
    except Exception as e:
        logger.error("Startup failed", error=str(e), exc_info=True)
        sys.exit(1)
    
    yield
    
    # Shutdown
    try:
        logger.info("Starting application shutdown")
        
        # Close HR Assistant
        await hr_assistant.close()
        
        # Close Redis
        if redis_client:
            await redis_client.close()
        
        logger.info("Application shutdown complete")
        
    except Exception as e:
        logger.error("Shutdown error", error=str(e))

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Production HR Assistant API with RAG capabilities",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    openapi_url="/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add middleware in correct order (outermost to innermost)

# Trusted Host middleware (security)
if settings.ENVIRONMENT == "production":
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*.yourdomain.com", "localhost"]
    )

# CORS middleware
if settings.ENABLE_CORS:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-RateLimit-Limit", "X-RateLimit-Remaining", "X-RateLimit-Reset"]
    )

# GZip compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Authentication middleware
app.add_middleware(AuthMiddleware)

# Rate limiting middleware
if settings.RATE_LIMIT_ENABLED:
    @app.on_event("startup")
    async def setup_rate_limiting():
        rate_limit_middleware = RateLimitMiddleware(app, redis_client)
        app.add_middleware(lambda app: rate_limit_middleware)

# Prometheus metrics
if settings.METRICS_ENABLED:
    instrumentator = Instrumentator(
        should_group_status_codes=True,
        should_group_untemplated=True,
        should_respect_env_var=False,
        should_instrument_requests_inprogress=True,
        excluded_handlers=[".*health.*", ".*metrics.*"],
        inprogress_name="hr_assistant_requests_inprogress",
        inprogress_labels=True
    )
    instrumentator.instrument(app).expose(app, endpoint="/metrics")

# OpenTelemetry instrumentation
if settings.TRACING_ENABLED:
    FastAPIInstrumentor.instrument_app(app)

# Include routers
app.include_router(router, prefix=settings.API_PREFIX)
app.include_router(search_router, prefix=settings.API_PREFIX)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.ENVIRONMENT,
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors"""
    logger.error("Unhandled exception", 
                error=str(exc),
                path=request.url.path,
                method=request.method,
                exc_info=True)
    
    # Don't expose internal errors in production
    if settings.ENVIRONMENT == "production":
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "message": "An unexpected error occurred",
                "timestamp": datetime.utcnow().isoformat()
            }
        )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error": type(exc).__name__,
                "message": str(exc),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

# Request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add unique request ID for tracing"""
    import uuid
    request_id = str(uuid.uuid4())
    
    # Add to request state
    request.state.request_id = request_id
    
    # Add to logging context
    structlog.contextvars.bind_contextvars(request_id=request_id)
    
    # Process request
    response = await call_next(request)
    
    # Add to response headers
    response.headers["X-Request-ID"] = request_id
    
    # Clear context
    structlog.contextvars.clear_contextvars()
    
    return response

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all requests and responses"""
    start_time = datetime.utcnow()
    
    # Log request
    logger.info("Request received",
               method=request.method,
               path=request.url.path,
               client=request.client.host if request.client else None)
    
    # Process request
    response = await call_next(request)
    
    # Calculate duration
    duration = (datetime.utcnow() - start_time).total_seconds()
    
    # Log response
    logger.info("Request completed",
               method=request.method,
               path=request.url.path,
               status_code=response.status_code,
               duration_seconds=duration)
    
    return response
if __name__ == "__main__":
    import sys
    import logging
    import uvicorn

    # import settings
    from app.core.config import settings

    def _safe_level(val):
        # normalize log level to string name
        if val is None:
            return "INFO"
        if isinstance(val, int):
            return logging.getLevelName(val)
        s = str(val)
        # if Enum passed, get its name
        if hasattr(val, "name"):
            s = val.name
        s = s.upper()
        valid = {"CRITICAL","FATAL","ERROR","WARNING","WARN","INFO","DEBUG","NOTSET"}
        return s if s in valid else "INFO"

    LOG_LEVEL = _safe_level(settings.LOG_LEVEL)

    # Add required uvicorn formatters/handlers/loggers. include json formatter if available and desired.
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {"format": settings.LOG_FORMAT},
            "access": {
                "()": "uvicorn.logging.AccessFormatter",
                "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s'
            },
            "uvicorn_default": {
                "()": "uvicorn.logging.DefaultFormatter",
                "fmt": "%(levelprefix)s %(message)s",
                "use_colors": False,
            },
        },
        "handlers": {
            "default": {
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "access": {
                "formatter": "access",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
        },
        "loggers": {
            "uvicorn": {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.error": {"handlers": ["default"], "level": LOG_LEVEL, "propagate": False},
            "uvicorn.access": {"handlers": ["access"], "level": LOG_LEVEL, "propagate": False},
        },
        "root": {"level": LOG_LEVEL, "handlers": ["default"]},
    }

    # Add JSON formatter if package available and user wants JSON
    if settings.LOG_JSON_FORMAT:
        try:
            from pythonjsonlogger import jsonlogger  # noqa: F401
            LOGGING_CONFIG["formatters"]["json"] = {
                "()": "pythonjsonlogger.jsonlogger.JsonFormatter",
                "format": "%(asctime)s %(name)s %(levelname)s %(message)s"
            }
            LOGGING_CONFIG["handlers"]["default"]["formatter"] = "json"
        except Exception:
            # if JSON lib not available, keep plain 'default'
            pass

    # Final safety check
    try:
        # try starting uvicorn with our config
        uvicorn.run(
            "main_production:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            loop="uvloop",
            log_config=LOGGING_CONFIG,
            access_log=True,
            use_colors=False,
            server_header=False,
            date_header=False,
            limit_concurrency=1000,
            timeout_keep_alive=5,
            ssl_keyfile=None,
            ssl_certfile=None
        )
    except Exception as e:
        # fallback: print and run uvicorn defaults
        print("Failed starting uvicorn with custom logging config:", e, file=sys.stderr)
        print("Falling back to uvicorn default logging (log_config=None).", file=sys.stderr)
        uvicorn.run(
            "main_production:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            loop="uvloop",
            log_config=None,
            access_log=True,
            use_colors=False,
            server_header=False,
            date_header=False,
            limit_concurrency=1000,
            timeout_keep_alive=5,
            ssl_keyfile=None,
            ssl_certfile=None
        )
"""
Production Translation Service with External API Integration
Supports multiple translation providers with fallback
"""

import asyncio
import logging
import hashlib
from typing import Optional, Dict, List
from datetime import datetime, timedelta

import aiohttp
import deepl
from prometheus_client import Counter
import redis.asyncio as redis

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
translation_requests = Counter('translation_requests_total', 'Total translation requests', ['provider', 'status'])
translation_cache_hits = Counter('translation_cache_hits_total', 'Translation cache hits')

class ProductionTranslatorService:
    """
    Production translation service with multiple provider support
    """
    
    def __init__(self):
        self.deepl_client: Optional[deepl.Translator] = None
        self.google_translator: Optional[GoogleTranslator] = None
        self.session: Optional[aiohttp.ClientSession] = None
        self.redis_client: Optional[redis.Redis] = None
        self.cache: Dict[str, str] = {}
        
    async def initialize(self):
        """Initialize translation service"""
        try:
            # Initialize HTTP session
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.TRANSLATION_TIMEOUT_SECONDS)
            )
            
            # Initialize provider based on configuration
            if settings.TRANSLATION_PROVIDER == "deepl" and settings.TRANSLATION_API_KEY:
                self.deepl_client = deepl.Translator(settings.TRANSLATION_API_KEY)
                logger.info("DeepL translator initialized")
                
            elif settings.TRANSLATION_PROVIDER == "google":
                self.google_translator = GoogleTranslator()
                logger.info("Google translator initialized")
            
            # Initialize cache if enabled
            if settings.TRANSLATION_CACHE_ENABLED and settings.CACHE_ENABLED:
                self.redis_client = redis.Redis(
                    connection_pool=redis.ConnectionPool.from_url(
                        settings.REDIS_URL,
                        password=settings.REDIS_PASSWORD,
                        ssl=settings.REDIS_SSL,
                        max_connections=10
                    )
                )
                logger.info("Translation cache initialized")
                
        except Exception as e:
            logger.error(f"Failed to initialize translator: {str(e)}")
            raise
    
    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        if self.redis_client:
            await self.redis_client.close()
    
    async def detect_language_async(self, text: str) -> str:
        """
        Detect language of text asynchronously
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code
        """
        if not text:
            return "unknown"
        
        try:
            # Use provider-specific detection
            if self.deepl_client:
                # DeepL doesn't have separate language detection, use heuristics
                return self._detect_language_heuristic(text)
            
            elif self.google_translator:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    None,
                    self.google_translator.detect,
                    text
                )
                return result.lang
            
            else:
                # Fallback to heuristic detection
                return self._detect_language_heuristic(text)
                
        except Exception as e:
            logger.warning(f"Language detection failed: {str(e)}")
            return self._detect_language_heuristic(text)
    
    def _detect_language_heuristic(self, text: str) -> str:
        """Heuristic language detection based on character ranges"""
        # Japanese
        if any('\u3040' <= c <= '\u309F' or  # Hiragana
               '\u30A0' <= c <= '\u30FF' or  # Katakana  
               '\u4E00' <= c <= '\u9FFF'     # Kanji
               for c in text):
            return 'ja'
        
        # Chinese (simplified/traditional)
        if any('\u4E00' <= c <= '\u9FFF' for c in text) and \
           not any('\u3040' <= c <= '\u30FF' for c in text):
            return 'zh'
        
        # Korean
        if any('\uAC00' <= c <= '\uD7AF' for c in text):
            return 'ko'
        
        # Arabic
        if any('\u0600' <= c <= '\u06FF' or
               '\u0750' <= c <= '\u077F' or
               '\u08A0' <= c <= '\u08FF'
               for c in text):
            return 'ar'
        
        # Cyrillic
        if any('\u0400' <= c <= '\u04FF' for c in text):
            return 'ru'
        
        # Spanish/Portuguese (check for specific characters)
        if any(c in 'áéíóúñ¿¡' for c in text.lower()):
            return 'es'
        
        # French
        if any(c in 'àâçèéêëîïôùûœ' for c in text.lower()):
            return 'fr'
        
        # German
        if any(c in 'äöüß' for c in text.lower()):
            return 'de'
        
        # Default to English
        return 'en'
    
    async def translate_async(self,
                             text: str,
                             source_lang: Optional[str] = None,
                             target_lang: str = 'en') -> str:
        """
        Translate text asynchronously with caching and fallback
        
        Args:
            text: Text to translate
            source_lang: Source language (auto-detect if None)
            target_lang: Target language
            
        Returns:
            Translated text
        """
        if not text:
            return ""
        
        # Auto-detect source if not provided
        if not source_lang:
            source_lang = await self.detect_language_async(text)
        
        # Skip if already in target language
        if source_lang == target_lang:
            return text
        
        # Check cache
        cached = await self._get_cached_translation(text, source_lang, target_lang)
        if cached:
            translation_cache_hits.inc()
            return cached
        
        # Perform translation with fallback
        translated = None
        
        # Try primary provider
        try:
            if settings.TRANSLATION_PROVIDER == "deepl" and self.deepl_client:
                translated = await self._translate_deepl(text, source_lang, target_lang)
                translation_requests.labels(provider='deepl', status='success').inc()
                
            elif settings.TRANSLATION_PROVIDER == "google" and self.google_translator:
                translated = await self._translate_google(text, source_lang, target_lang)
                translation_requests.labels(provider='google', status='success').inc()
                
            elif settings.TRANSLATION_PROVIDER == "azure":
                translated = await self._translate_azure(text, source_lang, target_lang)
                translation_requests.labels(provider='azure', status='success').inc()
                
            elif settings.TRANSLATION_PROVIDER == "custom":
                translated = await self._translate_custom(text, source_lang, target_lang)
                translation_requests.labels(provider='custom', status='success').inc()
                
        except Exception as e:
            logger.warning(f"Primary translation failed: {str(e)}")
            translation_requests.labels(provider=settings.TRANSLATION_PROVIDER, status='error').inc()
        
        # Fallback to Google Translate if primary fails
        if not translated and settings.TRANSLATION_PROVIDER != "google":
            try:
                if not self.google_translator:
                    self.google_translator = GoogleTranslator()
                
                translated = await self._translate_google(text, source_lang, target_lang)
                translation_requests.labels(provider='google_fallback', status='success').inc()
                
            except Exception as e:
                logger.error(f"Fallback translation failed: {str(e)}")
                translation_requests.labels(provider='google_fallback', status='error').inc()
        
        # If all translation attempts fail, return marked original
        if not translated:
            translated = f"[Translation unavailable from {source_lang}] {text}"
        
        # Cache the translation
        await self._cache_translation(text, source_lang, target_lang, translated)
        
        return translated
    
    async def _translate_deepl(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using DeepL API"""
        loop = asyncio.get_event_loop()
        
        # Map language codes to DeepL format
        deepl_source = source_lang.upper() if source_lang != 'auto' else None
        deepl_target = target_lang.upper()
        
        # Handle special cases
        if deepl_target == 'EN':
            deepl_target = 'EN-US'  # or EN-GB based on preference
        
        result = await loop.run_in_executor(
            None,
            self.deepl_client.translate_text,
            text,
            source_lang=deepl_source,
            target_lang=deepl_target
        )
        
        return result.text
    
    async def _translate_google(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Google Translate"""
        loop = asyncio.get_event_loop()
        
        result = await loop.run_in_executor(
            None,
            self.google_translator.translate,
            text,
            src=source_lang if source_lang != 'auto' else 'auto',
            dest=target_lang
        )
        
        return result.text
    
    async def _translate_azure(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using Azure Cognitive Services"""
        if not settings.TRANSLATION_ENDPOINT or not settings.TRANSLATION_API_KEY:
            raise ValueError("Azure translation requires TRANSLATION_ENDPOINT and TRANSLATION_API_KEY")
        
        headers = {
            'Ocp-Apim-Subscription-Key': settings.TRANSLATION_API_KEY,
            'Content-Type': 'application/json'
        }
        
        params = {
            'api-version': '3.0',
            'to': target_lang
        }
        
        if source_lang and source_lang != 'auto':
            params['from'] = source_lang
        
        body = [{'text': text}]
        
        async with self.session.post(
            f"{settings.TRANSLATION_ENDPOINT}/translate",
            headers=headers,
            params=params,
            json=body
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data[0]['translations'][0]['text']
            else:
                error = await response.text()
                raise Exception(f"Azure translation failed: {error}")
    
    async def _translate_custom(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate using custom endpoint"""
        if not settings.TRANSLATION_ENDPOINT:
            raise ValueError("Custom translation requires TRANSLATION_ENDPOINT")
        
        headers = {}
        if settings.TRANSLATION_API_KEY:
            headers['Authorization'] = f"Bearer {settings.TRANSLATION_API_KEY}"
        
        body = {
            'text': text,
            'source_language': source_lang,
            'target_language': target_lang
        }
        
        async with self.session.post(
            settings.TRANSLATION_ENDPOINT,
            headers=headers,
            json=body
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('translated_text', text)
            else:
                error = await response.text()
                raise Exception(f"Custom translation failed: {error}")
    
    def _generate_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        key_string = f"{source_lang}:{target_lang}:{text}"
        return f"translation:{hashlib.md5(key_string.encode()).hexdigest()}"
    
    async def _get_cached_translation(self, 
                                     text: str, 
                                     source_lang: str, 
                                     target_lang: str) -> Optional[str]:
        """Get cached translation if available"""
        if not settings.TRANSLATION_CACHE_ENABLED:
            return None
        
        cache_key = self._generate_cache_key(text, source_lang, target_lang)
        
        # Check in-memory cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Check Redis cache
        if self.redis_client:
            try:
                cached = await self.redis_client.get(cache_key)
                if cached:
                    translated = cached.decode('utf-8')
                    # Update in-memory cache
                    self._update_memory_cache(cache_key, translated)
                    return translated
            except Exception as e:
                logger.warning(f"Cache lookup failed: {str(e)}")
        
        return None
    
    async def _cache_translation(self,
                                text: str,
                                source_lang: str,
                                target_lang: str,
                                translated: str) -> None:
        """Cache translation result"""
        if not settings.TRANSLATION_CACHE_ENABLED:
            return
        
        cache_key = self._generate_cache_key(text, source_lang, target_lang)
        
        # Update in-memory cache
        self._update_memory_cache(cache_key, translated)
        
        # Update Redis cache
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    cache_key,
                    settings.CACHE_TTL_SECONDS,
                    translated.encode('utf-8')
                )
            except Exception as e:
                logger.warning(f"Failed to cache translation: {str(e)}")
    
    def _update_memory_cache(self, key: str, value: str) -> None:
        """Update in-memory cache with LRU eviction"""
        # Simple LRU: remove oldest if at capacity
        if len(self.cache) >= settings.TRANSLATION_MAX_CACHE_SIZE:
            # Remove first (oldest) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = value
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            # Test translation capability
            test_text = "Hello"
            translated = await self.translate_async(test_text, 'en', 'es')
            return len(translated) > 0
            
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            raise

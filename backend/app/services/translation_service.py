"""
Translation Service for multilingual support
Handles language detection and translation between languages
"""

import logging
from typing import Optional, List, Dict
import re

logger = logging.getLogger(__name__)

class TranslationService:
    """
    Service for handling language detection and translation
    Supports Japanese to English translation as primary use case
    """
    
    def __init__(self, translation_api_key: Optional[str] = None):
        """
        Initialize translation service
        
        Args:
            translation_api_key: API key for translation service (if using external API)
        """
        self.api_key = translation_api_key
        # In production, would initialize actual translation client here
        # e.g., Google Translate, DeepL, or custom model
        
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text
        
        Args:
            text: Text to analyze
            
        Returns:
            ISO 639-1 language code (e.g., 'en', 'ja', 'zh')
        """
        if not text:
            return 'unknown'
        
        # Check for Japanese characters
        if self._contains_japanese(text):
            return 'ja'
        
        # Check for Chinese characters
        if self._contains_chinese(text):
            return 'zh'
        
        # Check for Korean characters
        if self._contains_korean(text):
            return 'ko'
        
        # Check for Arabic
        if self._contains_arabic(text):
            return 'ar'
        
        # Check for Cyrillic (Russian, etc.)
        if self._contains_cyrillic(text):
            return 'ru'
        
        # Default to English
        return 'en'
    
    def translate(self, 
                 text: str, 
                 source_lang: Optional[str] = None,
                 target_lang: str = 'en') -> str:
        """
        Translate text from source language to target language
        
        Args:
            text: Text to translate
            source_lang: Source language code (auto-detect if None)
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if not text:
            return ""
        
        # Auto-detect source language if not provided
        if not source_lang:
            source_lang = self.detect_language(text)
        
        # Skip translation if already in target language
        if source_lang == target_lang:
            return text
        
        # In production, would call actual translation API here
        # For now, return a placeholder or simple translation
        
        try:
            # Simulate translation with actual API call placeholder
            translated = self._perform_translation(text, source_lang, target_lang)
            return translated
        except Exception as e:
            logger.error(f"Translation failed: {str(e)}")
            # Return original text if translation fails
            return text
    
    def translate_batch(self, 
                       texts: List[str],
                       source_lang: Optional[str] = None,
                       target_lang: str = 'en') -> List[str]:
        """
        Translate multiple texts in batch
        
        Args:
            texts: List of texts to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            List of translated texts
        """
        return [self.translate(text, source_lang, target_lang) for text in texts]
    
    def _contains_japanese(self, text: str) -> bool:
        """Check if text contains Japanese characters"""
        # Hiragana: \u3040-\u309F
        # Katakana: \u30A0-\u30FF
        # Kanji: \u4E00-\u9FFF
        return bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', text))
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters"""
        # CJK Unified Ideographs
        # Note: This overlaps with Japanese Kanji
        # In production, would use more sophisticated detection
        chinese_pattern = r'[\u4E00-\u9FFF]'
        # Check for absence of Hiragana/Katakana to distinguish from Japanese
        has_cjk = bool(re.search(chinese_pattern, text))
        has_kana = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text))
        return has_cjk and not has_kana
    
    def _contains_korean(self, text: str) -> bool:
        """Check if text contains Korean characters"""
        # Hangul: \uAC00-\uD7AF
        return bool(re.search(r'[\uAC00-\uD7AF]', text))
    
    def _contains_arabic(self, text: str) -> bool:
        """Check if text contains Arabic characters"""
        # Arabic: \u0600-\u06FF
        return bool(re.search(r'[\u0600-\u06FF]', text))
    
    def _contains_cyrillic(self, text: str) -> bool:
        """Check if text contains Cyrillic characters"""
        # Cyrillic: \u0400-\u04FF
        return bool(re.search(r'[\u0400-\u04FF]', text))
    
    def _perform_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Perform actual translation using translation service
        
        In production, this would integrate with:
        - Google Translate API
        - DeepL API
        - Azure Translator
        - Custom translation models
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        # Placeholder implementation
        # In production, replace with actual API call
        
        # Simple mock translations for demonstration
        mock_translations = {
            ('ja', 'en'): {
                '有給休暇': 'paid leave',
                '産休': 'maternity leave',
                '育児休暇': 'parental leave',
                '病気休暇': 'sick leave',
                '福利厚生': 'employee benefits',
                '退職金': 'retirement benefits',
                '健康保険': 'health insurance',
                '勤務時間': 'working hours',
                '残業': 'overtime',
                '給与': 'salary',
            }
        }
        
        # Check for exact matches in mock translations
        if (source_lang, target_lang) in mock_translations:
            translations_dict = mock_translations[(source_lang, target_lang)]
            for original, translated in translations_dict.items():
                if original in text:
                    text = text.replace(original, translated)
        
        # For demonstration, return text with translation marker
        # In production, this would be actual translated text
        if source_lang != target_lang and source_lang != 'en':
            return f"[Translated from {source_lang}] {text}"
        
        return text
    

class TranslationCache:
    """
    Cache for translation results to avoid redundant API calls
    """
    
    def __init__(self, max_size: int = 1000):
        """
        Initialize translation cache
        
        Args:
            max_size: Maximum number of cached translations
        """
        self.cache: Dict[str, str] = {}
        self.max_size = max_size
    
    def get_cache_key(self, text: str, source_lang: str, target_lang: str) -> str:
        """Generate cache key for translation"""
        return f"{source_lang}:{target_lang}:{hash(text)}"
    
    def get(self, text: str, source_lang: str, target_lang: str) -> Optional[str]:
        """Get cached translation if available"""
        key = self.get_cache_key(text, source_lang, target_lang)
        return self.cache.get(key)
    
    def set(self, text: str, source_lang: str, target_lang: str, translation: str) -> None:
        """Cache a translation"""
        # Implement simple LRU by removing oldest if at capacity
        if len(self.cache) >= self.max_size:
            # Remove first (oldest) item
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        key = self.get_cache_key(text, source_lang, target_lang)
        self.cache[key] = translation


class EnhancedTranslationService(TranslationService):
    """
    Enhanced translation service with caching and batch optimization
    """
    
    def __init__(self, translation_api_key: Optional[str] = None, use_cache: bool = True):
        """
        Initialize enhanced translation service
        
        Args:
            translation_api_key: API key for translation service
            use_cache: Whether to use translation caching
        """
        super().__init__(translation_api_key)
        self.cache = TranslationCache() if use_cache else None
    
    def translate(self, 
                 text: str, 
                 source_lang: Optional[str] = None,
                 target_lang: str = 'en') -> str:
        """
        Translate with caching support
        
        Args:
            text: Text to translate
            source_lang: Source language code
            target_lang: Target language code
            
        Returns:
            Translated text
        """
        if not text:
            return ""
        
        # Auto-detect source language if not provided
        if not source_lang:
            source_lang = self.detect_language(text)
        
        # Skip translation if already in target language
        if source_lang == target_lang:
            return text
        
        # Check cache first
        if self.cache:
            cached = self.cache.get(text, source_lang, target_lang)
            if cached:
                logger.debug(f"Translation cache hit for {source_lang}->{target_lang}")
                return cached
        
        # Perform translation
        translated = super().translate(text, source_lang, target_lang)
        
        # Cache the result
        if self.cache and translated != text:
            self.cache.set(text, source_lang, target_lang, translated)
        
        return translated

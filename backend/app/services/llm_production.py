"""
Production LLM Service with Multiple Provider Support
Handles LLM inference with retry, fallback, and monitoring
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime

import aiohttp
import openai
from anthropic import AsyncAnthropic
from prometheus_client import Counter, Histogram
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import settings

logger = logging.getLogger(__name__)

# Metrics
llm_requests = Counter('llm_requests_total', 'Total LLM requests', ['provider', 'status'])
llm_latency = Histogram('llm_latency_seconds', 'LLM response latency', ['provider'])
llm_tokens = Counter('llm_tokens_total', 'Total tokens processed', ['type'])

class ProductionLLMService:
    """
    Production LLM service with multiple provider support
    """
    
    def __init__(self):
        self.session: Optional[aiohttp.ClientSession] = None
        self.openai_client = None
        self.anthropic_client = None
        self.custom_endpoint = settings.LLM_ENDPOINT
        
    async def initialize(self):
        """Initialize LLM service"""
        try:
            # Initialize HTTP session for custom endpoints
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=settings.LLM_TIMEOUT_SECONDS)
            )
            
            # Initialize provider clients based on configuration
            if "gpt" in settings.LLM_MODEL_NAME.lower() and settings.LLM_API_KEY:
                openai.api_key = settings.LLM_API_KEY
                self.openai_client = openai
                logger.info("OpenAI client initialized")
                
            elif "claude" in settings.LLM_MODEL_NAME.lower() and settings.LLM_API_KEY:
                self.anthropic_client = AsyncAnthropic(api_key=settings.LLM_API_KEY)
                logger.info("Anthropic client initialized")
            
            logger.info(f"LLM service initialized with model: {settings.LLM_MODEL_NAME}")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM service: {str(e)}")
            raise
    
    async def close(self):
        """Cleanup resources"""
        if self.session:
            await self.session.close()
        
        if self.anthropic_client:
            await self.anthropic_client.close()
    
    @retry(
        stop=stop_after_attempt(settings.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=settings.LLM_RETRY_DELAY)
    )
    async def generate_async(self,
                            prompt: str,
                            temperature: float = 0.0,
                            max_tokens: int = 500,
                            system_prompt: Optional[str] = None) -> str:
        """
        Generate text using LLM with retry logic
        
        Args:
            prompt: User prompt
            temperature: Generation temperature (0.0 for deterministic)
            max_tokens: Maximum tokens to generate
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        try:
            # Route to appropriate provider
            if self.openai_client and "gpt" in settings.LLM_MODEL_NAME.lower():
                response = await self._generate_openai(
                    prompt, temperature, max_tokens, system_prompt
                )
                provider = "openai"
                
            elif self.anthropic_client and "claude" in settings.LLM_MODEL_NAME.lower():
                response = await self._generate_anthropic(
                    prompt, temperature, max_tokens, system_prompt
                )
                provider = "anthropic"
                
            else:
                # Use custom endpoint
                response = await self._generate_custom(
                    prompt, temperature, max_tokens, system_prompt
                )
                provider = "custom"
            
            # Record metrics
            latency = time.time() - start_time
            llm_latency.labels(provider=provider).observe(latency)
            llm_requests.labels(provider=provider, status='success').inc()
            
            # Estimate tokens (rough approximation)
            input_tokens = len(prompt.split()) * 1.3
            output_tokens = len(response.split()) * 1.3
            llm_tokens.labels(type='input').inc(input_tokens)
            llm_tokens.labels(type='output').inc(output_tokens)
            
            return response
            
        except Exception as e:
            llm_requests.labels(provider='unknown', status='error').inc()
            logger.error(f"LLM generation failed: {str(e)}")
            raise
    
    async def _generate_openai(self,
                              prompt: str,
                              temperature: float,
                              max_tokens: int,
                              system_prompt: Optional[str]) -> str:
        """Generate using OpenAI API"""
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        # Use async completion
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: openai.ChatCompletion.create(
                model=settings.LLM_MODEL_NAME,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                n=1,
                stop=None
            )
        )
        
        return response.choices[0].message.content.strip()
    
    async def _generate_anthropic(self,
                                 prompt: str,
                                 temperature: float,
                                 max_tokens: int,
                                 system_prompt: Optional[str]) -> str:
        """Generate using Anthropic API"""
        messages = [{"role": "user", "content": prompt}]
        
        response = await self.anthropic_client.messages.create(
            model=settings.LLM_MODEL_NAME,
            messages=messages,
            system=system_prompt if system_prompt else None,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        return response.content[0].text.strip()
    
    async def _generate_custom(self,
                              prompt: str,
                              temperature: float,
                              max_tokens: int,
                              system_prompt: Optional[str]) -> str:
        """Generate using custom endpoint"""
        headers = {
            'Content-Type': 'application/json'
        }
        
        if settings.LLM_API_KEY:
            headers['Authorization'] = f"Bearer {settings.LLM_API_KEY}"
        
        # Build request body (adjust based on your custom endpoint)
        body = {
            'model': settings.LLM_MODEL_NAME,
            'prompt': prompt,
            'temperature': temperature,
            'max_tokens': max_tokens,
            'stream': False
        }
        
        if system_prompt:
            body['system_prompt'] = system_prompt
        
        async with self.session.post(
            f"{self.custom_endpoint}/generate",
            headers=headers,
            json=body
        ) as response:
            if response.status == 200:
                data = await response.json()
                return data.get('text', '').strip()
            else:
                error = await response.text()
                raise Exception(f"Custom LLM failed: {error}")
    
    async def generate_structured(self,
                                 prompt: str,
                                 schema: Dict[str, Any],
                                 temperature: float = 0.0) -> Dict[str, Any]:
        """
        Generate structured output following a schema
        
        Args:
            prompt: Generation prompt
            schema: Expected output schema
            temperature: Generation temperature
            
        Returns:
            Structured response matching schema
        """
        # Add schema instructions to prompt
        structured_prompt = f"""{prompt}

You must respond with valid JSON that matches this schema:
{json.dumps(schema, indent=2)}

Response (JSON only):"""
        
        response = await self.generate_async(
            structured_prompt,
            temperature=temperature,
            max_tokens=settings.LLM_MAX_TOKENS
        )
        
        # Parse and validate response
        try:
            import json
            parsed = json.loads(response)
            
            # Basic schema validation
            for key in schema.keys():
                if key not in parsed:
                    logger.warning(f"Missing required field in LLM response: {key}")
                    parsed[key] = None
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse structured LLM response: {str(e)}")
            # Return empty structure matching schema
            return {key: None for key in schema.keys()}
    
    async def batch_generate(self,
                           prompts: List[str],
                           temperature: float = 0.0,
                           max_tokens: int = 500) -> List[str]:
        """
        Generate responses for multiple prompts in parallel
        
        Args:
            prompts: List of prompts
            temperature: Generation temperature
            max_tokens: Max tokens per response
            
        Returns:
            List of generated responses
        """
        # Process prompts in parallel with concurrency limit
        semaphore = asyncio.Semaphore(5)  # Limit to 5 concurrent requests
        
        async def generate_with_semaphore(prompt):
            async with semaphore:
                try:
                    return await self.generate_async(prompt, temperature, max_tokens)
                except Exception as e:
                    logger.error(f"Batch generation failed for prompt: {str(e)}")
                    return ""
        
        tasks = [generate_with_semaphore(prompt) for prompt in prompts]
        responses = await asyncio.gather(*tasks)
        
        return responses
    
    async def stream_generate(self,
                            prompt: str,
                            temperature: float = 0.0,
                            max_tokens: int = 500) -> AsyncIterator[str]:
        """
        Stream generated text token by token
        
        Args:
            prompt: Generation prompt
            temperature: Generation temperature
            max_tokens: Maximum tokens
            
        Yields:
            Generated text chunks
        """
        # This would implement streaming for supported providers
        # For now, return the full response as a single chunk
        response = await self.generate_async(prompt, temperature, max_tokens)
        yield response
    
    async def health_check(self) -> bool:
        """Check service health"""
        try:
            # Test generation with simple prompt
            test_prompt = "Respond with 'OK' if you are functioning."
            response = await self.generate_async(
                test_prompt,
                temperature=0.0,
                max_tokens=10
            )
            
            return len(response) > 0
            
        except Exception as e:
            logger.error(f"LLM health check failed: {str(e)}")
            raise

# Import AsyncIterator for type hints
from typing import AsyncIterator

import json

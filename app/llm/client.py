import asyncio
import logging
import random
import time
from typing import Any, Optional, Type
from pydantic import BaseModel

from app.core.config import settings
from app.llm.cache import ResponseCache
from app.llm.providers.base import BaseLLMProvider
from app.llm.providers.gemini_provider import GeminiProvider
from app.llm.providers.groq_provider import GroqProvider

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified LLM Client providing:
    - Provider abstraction (Gemini + Groq)
    - Automatic fallback cascade on failure or 429 rate limit
    - Exponential backoff with jitter
    - Deterministic prompt response caching
    """

    def __init__(
        self,
        primary_provider: Optional[BaseLLMProvider] = None,
        fallback_provider: Optional[BaseLLMProvider] = None,
        cache: Optional[ResponseCache] = None,
    ):
        self.primary = primary_provider or GeminiProvider()
        self.fallback = fallback_provider or GroqProvider()
        self.cache = cache or ResponseCache()

    def _execute_with_retry(
        self,
        provider: BaseLLMProvider,
        prompt: str,
        system_instruction: Optional[str],
        model: Optional[str],
        temperature: float,
        max_tokens: Optional[int],
        response_schema: Optional[Type[BaseModel] | dict[str, Any]],
        max_retries: int = 2,
    ) -> str:
        """Execute call with exponential backoff on transient errors."""
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return provider.generate(
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                )
            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                is_rate_limit = "429" in err_str or "quota" in err_str or "resourceexhausted" in err_str
                if is_rate_limit and attempt < max_retries:
                    sleep_time = (2 ** attempt) + random.uniform(0.1, 0.5)
                    logger.warning(
                        f"[{provider.provider_name}] Rate limited (attempt {attempt + 1}/{max_retries + 1}). "
                        f"Retrying in {sleep_time:.2f}s..."
                    )
                    time.sleep(sleep_time)
                else:
                    break

        raise last_error or RuntimeError(f"Unknown error from provider {provider.provider_name}")

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        use_cache: bool = True,
    ) -> str:
        """
        Synchronously generate text with caching, retry, and fallback.
        """
        effective_model = model or (settings.PRIMARY_LLM_MODEL if provider_name != "groq" else settings.FALLBACK_LLM_MODEL)
        
        # 1. Check cache
        cache_key = ""
        if use_cache:
            schema_dict = response_schema.model_json_schema() if (isinstance(response_schema, type) and issubclass(response_schema, BaseModel)) else response_schema
            cache_key = self.cache.compute_key(
                prompt=prompt,
                system_instruction=system_instruction,
                model=effective_model,
                temperature=temperature,
                response_schema=schema_dict,
            )
            cached_val = self.cache.get(cache_key)
            if cached_val is not None:
                logger.info(f"LLM cache hit for key {cache_key[:8]}...")
                return cached_val

        # 2. Select provider order
        if provider_name == "gemini":
            providers_to_try = [self.primary]
        elif provider_name == "groq":
            providers_to_try = [self.fallback]
        else:
            # Default auto-cascade
            providers_to_try = []
            if self.primary.is_available():
                providers_to_try.append(self.primary)
            if self.fallback.is_available():
                providers_to_try.append(self.fallback)

        if not providers_to_try:
            raise RuntimeError(
                "No LLM providers are configured with valid API keys. "
                "Please configure GEMINI_API_KEY or GROQ_API_KEY in .env"
            )

        last_err = None
        for prov in providers_to_try:
            try:
                prov_model = model
                if not prov_model:
                    prov_model = settings.PRIMARY_LLM_MODEL if prov.provider_name == "gemini" else settings.FALLBACK_LLM_MODEL
                
                logger.info(f"Calling LLM provider '{prov.provider_name}' with model '{prov_model}'")
                result = self._execute_with_retry(
                    provider=prov,
                    prompt=prompt,
                    system_instruction=system_instruction,
                    model=prov_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_schema=response_schema,
                )
                
                # Cache successful response
                if use_cache and cache_key:
                    self.cache.set(cache_key, result)

                return result
            except Exception as e:
                logger.warning(f"Provider '{prov.provider_name}' failed with error: {e}. Attempting fallback...")
                last_err = e

        raise RuntimeError(f"All configured LLM providers failed. Last error: {last_err}")

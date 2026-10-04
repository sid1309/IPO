import logging
from typing import Any, Optional, Type
from pydantic import BaseModel
from app.core.config import settings
from app.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GeminiProvider(BaseLLMProvider):
    """Adapter for Google Gemini API via official google-genai SDK."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.GEMINI_API_KEY
        self._client = None
        if self._api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini client: {e}")

    @property
    def provider_name(self) -> str:
        return "gemini"

    def is_available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def _build_config(
        self,
        system_instruction: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    ):
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "temperature": temperature,
        }
        if max_tokens:
            config_kwargs["max_output_tokens"] = max_tokens
        if system_instruction:
            config_kwargs["system_instruction"] = system_instruction
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
                config_kwargs["response_schema"] = response_schema
            elif isinstance(response_schema, dict):
                config_kwargs["response_schema"] = response_schema

        return types.GenerateContentConfig(**config_kwargs)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    ) -> str:
        if not self.is_available():
            raise RuntimeError("GeminiProvider is not configured with an API key.")

        model_name = model or settings.PRIMARY_LLM_MODEL
        config = self._build_config(system_instruction, temperature, max_tokens, response_schema)

        response = self._client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

    async def generate_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    ) -> str:
        if not self.is_available():
            raise RuntimeError("GeminiProvider is not configured with an API key.")

        model_name = model or settings.PRIMARY_LLM_MODEL
        config = self._build_config(system_instruction, temperature, max_tokens, response_schema)

        response = await self._client.aio.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        return response.text or ""

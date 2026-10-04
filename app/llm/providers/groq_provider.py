import logging
from typing import Any, Optional, Type
from pydantic import BaseModel
from app.core.config import settings
from app.llm.providers.base import BaseLLMProvider

logger = logging.getLogger(__name__)

class GroqProvider(BaseLLMProvider):
    """Adapter for Groq API (Llama 3.3 70B, Llama 3.1 8B)."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or settings.GROQ_API_KEY
        self._client = None
        self._async_client = None
        if self._api_key:
            try:
                from groq import Groq, AsyncGroq
                self._client = Groq(api_key=self._api_key)
                self._async_client = AsyncGroq(api_key=self._api_key)
            except Exception as e:
                logger.warning(f"Failed to initialize Groq client: {e}")

    @property
    def provider_name(self) -> str:
        return "groq"

    def is_available(self) -> bool:
        return self._client is not None and bool(self._api_key)

    def _build_messages(self, prompt: str, system_instruction: Optional[str] = None) -> list[dict[str, str]]:
        messages = []
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})
        messages.append({"role": "user", "content": prompt})
        return messages

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
            raise RuntimeError("GroqProvider is not configured with an API key.")

        model_name = model or settings.FALLBACK_LLM_MODEL
        messages = self._build_messages(prompt, system_instruction)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

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
            raise RuntimeError("GroqProvider is not configured with an API key.")

        model_name = model or settings.FALLBACK_LLM_MODEL
        messages = self._build_messages(prompt, system_instruction)

        kwargs: dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            kwargs["max_tokens"] = max_tokens
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self._async_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

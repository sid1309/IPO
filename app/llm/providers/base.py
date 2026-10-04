from abc import ABC, abstractmethod
from typing import Any, Optional, Type
from pydantic import BaseModel

class BaseLLMProvider(ABC):
    """Abstract base class for all LLM provider adapters."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Name of the provider, e.g. 'gemini' or 'groq'."""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider is configured with valid credentials."""
        pass

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    ) -> str:
        """Generate text completion synchronously."""
        pass

    @abstractmethod
    async def generate_async(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        response_schema: Optional[Type[BaseModel] | dict[str, Any]] = None,
    ) -> str:
        """Generate text completion asynchronously."""
        pass

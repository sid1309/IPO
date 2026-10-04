import pytest
from unittest.mock import MagicMock
from app.core.config import settings
from app.llm.cache import ResponseCache
from app.llm.client import LLMClient
from app.llm.providers.base import BaseLLMProvider

class MockProvider(BaseLLMProvider):
    def __init__(self, name: str, should_fail: bool = False, response: str = "mock output"):
        self._name = name
        self.should_fail = should_fail
        self.response = response
        self.call_count = 0

    @property
    def provider_name(self) -> str:
        return self._name

    def is_available(self) -> bool:
        return True

    def generate(self, prompt: str, **kwargs) -> str:
        self.call_count += 1
        if self.should_fail:
            raise RuntimeError(f"Simulated 429 rate limit in {self._name}")
        return self.response

    async def generate_async(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


def test_config_loads_and_ensures_dirs():
    assert settings.PROJECT_NAME == "IPO Prospectus Analyst"
    assert settings.PRIMARY_LLM_MODEL == "gemini-3.8-flash"
    assert settings.FALLBACK_LLM_MODEL == "qwen/qwen3.8-27b"


def test_response_cache(tmp_path):
    cache = ResponseCache(cache_dir=str(tmp_path))
    key = cache.compute_key(prompt="test prompt", model="gemini-2.5-flash")
    
    assert cache.get(key) is None
    cache.set(key, "cached answer")
    assert cache.get(key) == "cached answer"


def test_llm_client_primary_success(tmp_path):
    cache = ResponseCache(cache_dir=str(tmp_path))
    primary = MockProvider("gemini", should_fail=False, response="Gemini Answer")
    fallback = MockProvider("groq", should_fail=False, response="Groq Answer")

    client = LLMClient(primary_provider=primary, fallback_provider=fallback, cache=cache)
    resp = client.generate("Hello IPO", use_cache=False)

    assert resp == "Gemini Answer"
    assert primary.call_count == 1
    assert fallback.call_count == 0


def test_llm_client_fallback_on_primary_failure(tmp_path):
    cache = ResponseCache(cache_dir=str(tmp_path))
    primary = MockProvider("gemini", should_fail=True)
    fallback = MockProvider("groq", should_fail=False, response="Groq Fallback Answer")

    client = LLMClient(primary_provider=primary, fallback_provider=fallback, cache=cache)
    resp = client.generate("Hello IPO", use_cache=False)

    # Primary should fail, client should automatically cascade to fallback
    assert resp == "Groq Fallback Answer"
    assert primary.call_count >= 1
    assert fallback.call_count == 1


def test_llm_client_caching_behavior(tmp_path):
    cache = ResponseCache(cache_dir=str(tmp_path))
    primary = MockProvider("gemini", should_fail=False, response="Cached Result")
    fallback = MockProvider("groq", should_fail=False)

    client = LLMClient(primary_provider=primary, fallback_provider=fallback, cache=cache)

    # First call: hits provider and caches
    r1 = client.generate("What is the fresh issue?", use_cache=True)
    assert r1 == "Cached Result"
    assert primary.call_count == 1

    # Second call: identical parameters should hit cache and NOT invoke provider
    r2 = client.generate("What is the fresh issue?", use_cache=True)
    assert r2 == "Cached Result"
    assert primary.call_count == 1  # Call count remains 1!

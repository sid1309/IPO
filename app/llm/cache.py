import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Optional
from app.core.config import settings

logger = logging.getLogger(__name__)

class ResponseCache:
    """Disk-backed cache for LLM completions to save quota and accelerate testing."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR) / "llm_responses"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            import diskcache
            self._disk_cache = diskcache.Cache(str(self.cache_dir))
            self._use_diskcache = True
        except ImportError:
            self._disk_cache = None
            self._use_diskcache = False
            self._memory_cache: dict[str, str] = {}

    @staticmethod
    def compute_key(
        prompt: str,
        system_instruction: Optional[str] = None,
        model: str = "",
        temperature: float = 0.0,
        response_schema: Optional[dict[str, Any]] = None,
    ) -> str:
        """Create a deterministic SHA-256 key from request parameters."""
        payload = {
            "prompt": prompt,
            "system_instruction": system_instruction or "",
            "model": model,
            "temperature": round(temperature, 4),
            "response_schema": response_schema,
        }
        serialized = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def get(self, cache_key: str) -> Optional[str]:
        """Retrieve cached text if present."""
        if self._use_diskcache and self._disk_cache is not None:
            return self._disk_cache.get(cache_key)
        return getattr(self, "_memory_cache", {}).get(cache_key)

    def set(self, cache_key: str, value: str, expire: Optional[int] = 86400 * 7) -> None:
        """Store text in cache (default TTL 7 days)."""
        if self._use_diskcache and self._disk_cache is not None:
            self._disk_cache.set(cache_key, value, expire=expire)
        else:
            if not hasattr(self, "_memory_cache"):
                self._memory_cache = {}
            self._memory_cache[cache_key] = value

    def clear(self) -> None:
        """Clear all cached responses."""
        if self._use_diskcache and self._disk_cache is not None:
            self._disk_cache.clear()
        if hasattr(self, "_memory_cache"):
            self._memory_cache.clear()

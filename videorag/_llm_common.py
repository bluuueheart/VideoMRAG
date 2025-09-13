import os
from dataclasses import dataclass

# Optional provider imports (moved from _llm.py for reuse)
try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI, APIConnectionError, RateLimitError  # type: ignore
    _OPENAI_AVAILABLE = True
except Exception:
    AsyncOpenAI = None  # type: ignore
    AsyncAzureOpenAI = None  # type: ignore
    APIConnectionError = Exception  # type: ignore
    RateLimitError = Exception  # type: ignore
    _OPENAI_AVAILABLE = False

# Defer importing ollama until actually needed to avoid import-time failure
AsyncClient = None  # type: ignore
_OLLAMA_AVAILABLE = None

from ._utils import wrap_embedding_func_with_attrs
from ._utils import EmbeddingFunc

# Global cached client instances (moved)
global_openai_async_client = None
global_azure_openai_async_client = None
global_custom_openai_async_client = None
global_ollama_client = None


def get_openai_async_client_instance():
    global global_openai_async_client
    if global_openai_async_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client is not available. Please install 'openai' and set OPENAI_API_KEY.")
        global_openai_async_client = AsyncOpenAI()
    return global_openai_async_client


def get_azure_openai_async_client_instance():
    global global_azure_openai_async_client
    if global_azure_openai_async_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("Azure OpenAI client is not available. Please install 'openai' and set Azure credentials.")
        global_azure_openai_async_client = AsyncAzureOpenAI()
    return global_azure_openai_async_client


def get_custom_openai_async_client_instance(base_url: str, api_key: str):
    global global_custom_openai_async_client
    if global_custom_openai_async_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client is not available. Please install 'openai'.")
        global_custom_openai_async_client = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return global_custom_openai_async_client


def get_external_llm_async_client_instance():
    global global_ollama_client
    if global_ollama_client is None:
        # Try to import an external LLM client lazily (Ollama-compatible)
        global _OLLAMA_AVAILABLE, AsyncClient
        if _OLLAMA_AVAILABLE is None:
            try:
                from ollama import AsyncClient as _AsyncClient  # type: ignore
                AsyncClient = _AsyncClient
                _OLLAMA_AVAILABLE = True
            except Exception:
                AsyncClient = None
                _OLLAMA_AVAILABLE = False

        if not _OLLAMA_AVAILABLE:
            raise RuntimeError("External LLM client not available. Configure a local HF model shortname via OLLAMA_CHAT_MODEL or install a supported external client.")
        global_ollama_client = AsyncClient()  # Adjust base URL if necessary
    return global_ollama_client

# Backwards-compatible alias
def get_ollama_async_client_instance():
    return get_external_llm_async_client_instance()


# Setup LLM Configuration (moved unchanged)
@dataclass
class LLMConfig:
    # To be set
    embedding_func_raw: callable
    embedding_model_name: str
    embedding_dim: int
    embedding_max_token_size: int
    embedding_batch_num: int
    embedding_func_max_async: int
    query_better_than_threshold: float

    best_model_func_raw: callable
    best_model_name: str
    best_model_max_token_size: int
    best_model_max_async: int

    cheap_model_func_raw: callable
    cheap_model_name: str
    cheap_model_max_token_size: int
    cheap_model_max_async: int

    # Assigned in post init
    embedding_func: EmbeddingFunc = None
    best_model_func: callable = None
    cheap_model_func: callable = None

    def __post_init__(self):
        embedding_wrapper = wrap_embedding_func_with_attrs(
            embedding_dim=self.embedding_dim,
            max_token_size=self.embedding_max_token_size,
            model_name=self.embedding_model_name
        )
        self.embedding_func = embedding_wrapper(self.embedding_func_raw)
        # 强制统一采样参数（所有通道/模式）：temperature=0.1, top_p=1
        def _force_sampling_kwargs(kw: dict | None) -> dict:
            kw = dict(kw or {})
            # 无条件覆盖，确保固定
            kw["temperature"] = 0.1
            kw["top_p"] = 1
            return kw

        self.best_model_func = lambda prompt, *a, **kw: self.best_model_func_raw(
            self.best_model_name, prompt, *a, **_force_sampling_kwargs(kw)
        )
        self.cheap_model_func = lambda prompt, *a, **kw: self.cheap_model_func_raw(
            self.cheap_model_name, prompt, *a, **_force_sampling_kwargs(kw)
        )

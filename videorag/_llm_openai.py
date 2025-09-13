import numpy as np
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import BaseKVStorage
from ._utils import compute_args_hash
from ._llm_common import (
    get_openai_async_client_instance,
    get_custom_openai_async_client_instance,
    APIConnectionError,
    RateLimitError,
    LLMConfig,
)

# ================= OpenAI & Custom OpenAI =================
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    openai_async_client = get_openai_async_client_instance()
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def custom_openai_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], base_url=None, api_key=None, **kwargs
) -> str:
    openai_async_client = get_custom_openai_async_client_instance(base_url, api_key)
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    response = await openai_async_client.chat.completions.create(
        model=model, messages=messages, **kwargs
    )

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response.choices[0].message.content, "model": model}}
        )
        await hashing_kv.index_done_callback()
    return response.choices[0].message.content


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def custom_openai_embedding(model_name: str, texts: list[str], base_url: str, api_key: str) -> np.ndarray:
    openai_async_client = get_custom_openai_async_client_instance(base_url, api_key)
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


async def gpt_4o_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def gpt_4o_mini_complete(
        model_name, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs,
    )


async def custom_gpt_complete(
        model_name, prompt, system_prompt=None, history_messages=[], base_url=None, api_key=None, **kwargs
) -> str:
    # 统一固定采样参数
    kwargs.setdefault("temperature", 0.1)
    kwargs.setdefault("top_p", 1)
    return await custom_openai_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        base_url=base_url,
        api_key=api_key,
        **kwargs,
    )


@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def openai_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    openai_async_client = get_openai_async_client_instance()
    response = await openai_async_client.embeddings.create(
        model=model_name, input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])


openai_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_complete,
    best_model_name = "gpt-4o",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

openai_4o_mini_config = LLMConfig(
    embedding_func_raw = openai_embedding,
    embedding_model_name = "text-embedding-3-small",
    embedding_dim = 1536,
    embedding_max_token_size  = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,

    # LLM        
    best_model_func_raw = gpt_4o_mini_complete,
    best_model_name = "gpt-4o-mini",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
        
    cheap_model_func_raw = gpt_4o_mini_complete,
    cheap_model_name = "gpt-4o-mini",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

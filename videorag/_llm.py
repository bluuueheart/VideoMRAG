import numpy as np
import asyncio
from io import BytesIO
from PIL import Image
import base64
import torch
import re

from dataclasses import asdict, dataclass, field

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
import os

from ._utils import compute_args_hash, wrap_embedding_func_with_attrs
from .base import BaseKVStorage
from ._utils import EmbeddingFunc

from ._llm_common import (
    AsyncOpenAI,
    AsyncAzureOpenAI,
    APIConnectionError,
    RateLimitError,
    AsyncClient,
    _OPENAI_AVAILABLE,
    _OLLAMA_AVAILABLE,
    get_openai_async_client_instance,
    get_custom_openai_async_client_instance,
    get_ollama_async_client_instance,
    get_external_llm_async_client_instance,
    LLMConfig,
)
from ._llm_azure import (
    azure_openai_complete_if_cache,
    azure_gpt_4o_complete,
    azure_gpt_4o_mini_complete,
    azure_openai_embedding,
    azure_openai_config,
)

from ._llm_openai import (
    openai_complete_if_cache,
    custom_openai_complete_if_cache,
    custom_openai_embedding,
    gpt_4o_complete,
    gpt_4o_mini_complete,
    custom_gpt_complete,
    openai_embedding,
    openai_config,
    openai_4o_mini_config,
)

# ===== Default chat model name (can be overridden via env). Use short names to select local HF models =====
# Supported short names: 'llama', 'qwen', 'gemma', 'internvl', 'minicpm', 'internvl'
DEFAULT_OLLAMA_CHAT_MODEL = os.environ.get("OLLAMA_CHAT_MODEL", "qwen")
DEFAULT_OLLAMA_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "all-MiniLM-L6-v2")

# Mapping short names to local filesystem Hugging Face model paths (user-provided)
# Update these paths if needed; user-specified paths from the request are used here.
MODEL_NAME_TO_LOCAL_PATH = {
    "llama": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/meta-llama/Llama-3.2-11B-Vision-Instruct",
    "qwen": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/Qwen/Qwen2___5-7B-Instruct",
    "qwen3": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/Qwen/Qwen3-32B",
    "gemma": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/google/gemma-3-12b-it",
    "minicpm": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/openbmb/MiniCPM-V-4_5",
    "internvl": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/OpenGVLab/InternVL3_5-8B-HF",
    # embedding model
    "all-MiniLM-L6-v2": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/sentence-transformers/all-MiniLM-L6-v2",
    # YOLO-World detector binary
    "yolov8m-worldv2": "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/yolov8m-worldv2.pt",
}

# Cache for loaded HF text models and tokenizers
_HF_TEXT_MODELS = {}

_HF_EMBEDDING_MODELS = {}


def _resolve_model_path_for_shortname(short: str) -> str | None:
    """Return configured local model path for a short name, or None."""
    if not short:
        return None
    p = MODEL_NAME_TO_LOCAL_PATH.get(short)
    # If configured path exists on filesystem, return it. Otherwise treat as not configured.
    if p:
        try:
            # If it's a local path (absolute or relative), ensure it exists to avoid HF hub interpreting it as repo id
            if os.path.isabs(p) or os.path.exists(p):
                if os.path.exists(p):
                    return p
                # allow non-absolute but existing relative paths
                # fallthrough to warning
            else:
                # If path string provided but doesn't exist, warn and treat as not configured
                print(f"[LLM][Warning] Configured local model path for '{short}' does not exist: {p}")
                return None
        except Exception:
            # On any filesystem check error, conservatively return None
            print(f"[LLM][Warning] Unable to verify local model path for '{short}': {p}")
            return None
    # allow qwen3 alias
    if short.startswith("qwen3") and "qwen3" in MODEL_NAME_TO_LOCAL_PATH:
        return MODEL_NAME_TO_LOCAL_PATH.get("qwen3")
    return None


async def local_complete_router(model_short_name: str, prompt: str, system_prompt: str | None = None, images_base64: list | None = None, **kwargs) -> str:
    """Route to an appropriate local completion implementation depending on model short name.
    Preference: vllm (if available) -> transformers -> internvl_hf_complete (already implemented)
    """
    path = _resolve_model_path_for_shortname(model_short_name)
    if not path:
        raise RuntimeError(f"No usable local model path configured for '{model_short_name}' (path missing or not found).")

    # InternVL handled separately
    if model_short_name == "internvl":
        return await internvl_hf_complete(model_name=path, prompt=prompt, system_prompt=system_prompt, images_base64=images_base64, **kwargs)

    # Try vllm first
    try:
        from vllm import LLM
        def _vllm_sync():
            with LLM(model=path) as llm:
                gen = llm.generate(prompt if not system_prompt else f"{system_prompt}\n\n{prompt}", max_tokens=int(kwargs.get("max_new_tokens", kwargs.get("max_new_tokens", 512))), temperature=float(kwargs.get("temperature", 0.1)))
                for r in gen:
                    try:
                        return r.outputs[0].text
                    except Exception:
                        return str(r)
            return ""

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _vllm_sync)
    except Exception:
        # fall back to transformers
        pass

    # Transformers path
    def _transformers_sync():
        from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
        # Ensure we only call from_pretrained with a local files-only load when path is a local directory
        from transformers import __version__ as _tf_ver
        kwargs_tf = {"trust_remote_code": True}
        # If path exists on filesystem, prefer local_files_only to avoid HF hub repo id parsing
        if os.path.exists(path):
            kwargs_tf["local_files_only"] = True
        try:
            tokenizer = AutoTokenizer.from_pretrained(path, use_fast=False, **kwargs_tf)
            model = AutoModelForCausalLM.from_pretrained(path, device_map="auto", **kwargs_tf)
        except Exception as e:
            # Re-raise with clearer context
            raise RuntimeError(f"Failed to load local HF model at {path}: {e}") from e
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        out = pipe(prompt if not system_prompt else f"{system_prompt}\n\n{prompt}", max_new_tokens=int(kwargs.get("max_new_tokens", 512)), do_sample=False)
        return out[0].get("generated_text") or out[0].get("text") or ""

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _transformers_sync)

def get_default_ollama_chat_model() -> str:
    return DEFAULT_OLLAMA_CHAT_MODEL

def get_default_ollama_embed_model() -> str:
    return DEFAULT_OLLAMA_EMBED_MODEL

## (Common utilities and LLMConfig moved to _llm_common.py)

##### OpenAI Configuration
## (OpenAI related functions & configs moved to _llm_openai.py)

# Azure OpenAI helpers were moved to videorag/_llm_azure.py during refactor.


######  External LLM configuration (Ollama-compatible client)

async def external_llm_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # Note: defer initializing Ollama client until we know we need it

    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    # Pop images_base64 to avoid it being part of the hash
    images_base64: list[str] | None = kwargs.pop("images_base64", None)
    messages = []
    
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    
    # For Ollama, pass images via the message 'images' field rather than concatenating into text
    # If VLM acceleration is requested, attempt to convert images into a torch tensor batch
    vlm_flag = False
    try:
        vlm_flag = bool(kwargs.get('vlm_accel') or os.environ.get('VLM_ACCEL'))
    except Exception:
        vlm_flag = False

    if images_base64 and vlm_flag:
        try:
            import torch
            from torchvision import transforms

            tensors = []
            tf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ])
            for b64 in images_base64:
                s = b64
                if s.startswith("data:image"):
                    s = s.split(',', 1)[1]
                img = Image.open(BytesIO(base64.b64decode(s))).convert('RGB')
                t = tf(img)
                tensors.append(t)
            if tensors:
                batch = torch.stack(tensors, dim=0)
                # attach to kwargs so downstream clients may reuse
                kwargs['images_tensors'] = batch
                # Also provide stripped base64 list for backwards compatibility
                def _strip_prefix(b64: str) -> str:
                    if b64.startswith("data:image"):
                        try:
                            return b64.split(",", 1)[1]
                        except Exception:
                            return b64
                    return b64
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "images": [_strip_prefix(img_b64) for img_b64 in images_base64]
                }
            else:
                user_message = {"role": "user", "content": prompt}
        except Exception:
            # Fallback to original handling on any failure
            def _strip_prefix(b64: str) -> str:
                if b64.startswith("data:image"):
                    try:
                        return b64.split(",", 1)[1]
                    except Exception:
                        return b64
                return b64

            if images_base64:
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "images": [_strip_prefix(img_b64) for img_b64 in images_base64]
                }
            else:
                user_message = {"role": "user", "content": prompt}
    elif images_base64:
        # Ollama expects raw base64 strings (without data:image/... prefix) in some versions
        # Ensure we strip possible prefix if present
        def _strip_prefix(b64: str) -> str:
            if b64.startswith("data:image"):
                try:
                    return b64.split(",", 1)[1]
                except Exception:
                    return b64
            return b64

        user_message = {
            "role": "user",
            "content": prompt,
            "images": [_strip_prefix(img_b64) for img_b64 in images_base64]
        }
    else:
        user_message = {"role": "user", "content": prompt}
    
    messages.append(user_message)

    if hashing_kv is not None:
        # Note: hash does not include images for simplicity, assuming prompt is unique enough
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        # NOTE: I update here to avoid the if_cache_return["return"] is None
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # Send the request to external LLM client (unless routed to local HF)
    # 统一固定采样参数
    options = {
        "keep_alive": -1,
        "temperature": 0.1,
        "top_p": 1,
    }
    # If model is a short name that maps to local HF model, delegate to HF wrapper
    short = (model or "").split(":")[0].lower()
    if short in MODEL_NAME_TO_LOCAL_PATH:
        # Route to central local router which picks vllm/transformers/internvl as needed
        try:
            return await local_complete_router(short, prompt, system_prompt=system_prompt, images_base64=images_base64, **kwargs)
        except Exception as e:
            # fallback to external client if available
            print(f"[LLM-Router] Local complete failed for {short}: {e}. Falling back to external client if available.")

    # If we reach here, use Ollama client
    ollama_client = get_external_llm_async_client_instance()
    response = await ollama_client.chat(
        model=model,
        messages=messages,
        options=options
    )
    # print(messages)
    # print(response['message']['content'])

    
    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": response['message']['content'], "model": model}}
        )
        await hashing_kv.index_done_callback()

    return response['message']['content']


async def external_llm_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await external_llm_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

# Backwards-compatible alias
async def ollama_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await external_llm_complete(model_name, prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)

async def external_llm_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await external_llm_complete_if_cache(
        # "deepseek-r1:latest",  # For now select your model
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

# Backwards-compatible alias
async def ollama_mini_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await external_llm_mini_complete(model_name, prompt, system_prompt=system_prompt, history_messages=history_messages, **kwargs)

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def external_llm_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # Initialize the Ollama client
    # If short name maps to local embedding model, use HF embedding
    short = (model_name or "").split(":")[0].lower()
    if short in MODEL_NAME_TO_LOCAL_PATH:
        try:
            return await hf_local_embedding(MODEL_NAME_TO_LOCAL_PATH[short], texts)
        except Exception as e:
            print(f"[LLM-Router] HF local embedding failed for {short}: {e}. Falling back to external embedding service if available.")

    ollama_client = get_external_llm_async_client_instance()

    # Send the request to Ollama for embeddings
    response = await ollama_client.embed(
        model=model_name,  
        input=texts
    )

    # Extract embeddings from the response
    embeddings = response['embeddings']

    return np.array(embeddings)

# Backwards-compatible alias
async def ollama_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    return await external_llm_embedding(model_name, texts)


async def hf_local_text_complete(model_short_name: str, prompt: str, system_prompt: str | None = None, images_base64: list | None = None, **kwargs) -> str:
    """Minimal async wrapper that loads a local HF model and runs text generation in a thread."""
    model_path = MODEL_NAME_TO_LOCAL_PATH.get(model_short_name)
    if model_path is None:
        raise RuntimeError(f"No local path configured for model {model_short_name}")

    # Verify the configured model path actually exists. If not, raise a clear error to avoid
    # passing a filesystem path string to transformers which may treat it as a repo id.
    if not os.path.exists(model_path):
        raise RuntimeError(f"Configured local model path for '{model_short_name}' does not exist: {model_path}")

    # If this is the InternVL model, delegate to its dedicated implementation
    if model_short_name == "internvl":
        return await internvl_hf_complete(model_name=model_path, prompt=prompt, system_prompt=system_prompt, images_base64=images_base64, **kwargs)

    def _sync_generate():
        # Try vllm first for faster, memory-efficient inference if available
        try:
            from vllm import LLM
            using_vllm = True
        except Exception:
            using_vllm = False

        if using_vllm:
            try:
                from vllm import LLM
                # Use a short lived LLM instance for sync generation
                with LLM(model=model_path) as llm:
                    gen = llm.generate(prompt if not system_prompt else f"{system_prompt}\n\n{prompt}", max_tokens=int(kwargs.get("max_new_tokens", 512)), temperature=float(kwargs.get("temperature", 0.1)))
                    # collect first result
                    for r in gen:
                        try:
                            return r.outputs[0].text
                        except Exception:
                            # fallback to stringification
                            return str(r)
            except Exception as e:
                # vllm failed; fall through to transformers
                print(f"[HF-Router] vllm generation failed for {model_short_name}: {e}")

        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
        except Exception as e:
            raise RuntimeError("transformers and torch are required for local HF generation") from e

        # Load/cache model
        key = f"text::{model_path}"
        if key not in _HF_TEXT_MODELS:
            # Ensure local-only loading when path is local to avoid HF hub repo id interpretation
            tf_kwargs = {"trust_remote_code": True}
            if os.path.exists(model_path):
                tf_kwargs["local_files_only"] = True
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False, **tf_kwargs)
            model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", **tf_kwargs)
            _HF_TEXT_MODELS[key] = (tokenizer, model)
        else:
            tokenizer, model = _HF_TEXT_MODELS[key]

        full_prompt = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")
        out = pipe(full_prompt, max_new_tokens=int(kwargs.get("max_new_tokens", 512)), do_sample=False)
        text = out[0].get("generated_text") or out[0].get("text") or ""
        return text

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_generate)


async def hf_local_embedding(model_path: str, texts: list[str]) -> np.ndarray:
    """Minimal async wrapper to compute embeddings using sentence-transformers or HF encoder."""
    def _sync_embed():
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as _np
        except Exception as e:
            raise RuntimeError("sentence-transformers is required for local embeddings") from e

        key = f"embed::{model_path}"
        if key not in _HF_EMBEDDING_MODELS:
            model = SentenceTransformer(model_path)
            _HF_EMBEDDING_MODELS[key] = model
        else:
            model = _HF_EMBEDDING_MODELS[key]

        emb = model.encode(texts, convert_to_numpy=True)
        return emb

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_embed)

# External-client-backed configuration (keeps naming for backward compatibility)
external_llm_config = LLMConfig(
    embedding_func_raw = external_llm_embedding,
    embedding_model_name = DEFAULT_OLLAMA_EMBED_MODEL,
    embedding_dim = 768,
    embedding_max_token_size=8192,
    embedding_batch_num = 1,
    embedding_func_max_async = 1,
    query_better_than_threshold = 0.2,
    best_model_func_raw = external_llm_complete ,
    best_model_name = DEFAULT_OLLAMA_CHAT_MODEL, # use Qwen2.5-VL 7B as generator
    best_model_max_token_size = 32768,
    best_model_max_async  = 1,
    cheap_model_func_raw = external_llm_mini_complete,
    cheap_model_name = DEFAULT_OLLAMA_CHAT_MODEL,
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 1
)

# Backwards-compatible alias
ollama_config = external_llm_config

# Add a post-init to wrap model functions to accept images_base64
_original_external_llm_post_init = external_llm_config.__post_init__
def _external_llm_post_init_wrapper(self):
    _original_external_llm_post_init(self)
    
    original_best_model_func = self.best_model_func
    def _best_with_vlm(prompt, *args, **kwargs):
        # Ensure vlm flag is forwarded
        kwargs.setdefault('vlm_accel', getattr(self, 'vlm_accel', True))
        return original_best_model_func(prompt, *args, **kwargs)
    self.best_model_func = _best_with_vlm

    original_cheap_model_func = self.cheap_model_func
    def _cheap_with_vlm(prompt, *args, **kwargs):
        kwargs.setdefault('vlm_accel', getattr(self, 'vlm_accel', True))
        return original_cheap_model_func(prompt, *args, **kwargs)
    self.cheap_model_func = _cheap_with_vlm
external_llm_config.__post_init__ = _external_llm_post_init_wrapper.__get__(external_llm_config)


###### DeepSeek Configuration
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_complete_if_cache(
    model, prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
    # 使用DeepSeek API
    import httpx
    
    hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    if hashing_kv is not None:
        args_hash = compute_args_hash(model, messages)
        if_cache_return = await hashing_kv.get_by_id(args_hash)
        if if_cache_return is not None and if_cache_return["return"] is not None:
            return if_cache_return["return"]

    # DeepSeek API调用
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {os.environ.get('DEEPSEEK_API_KEY', 'sk-*******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": messages,
                # 统一固定采样参数
                "temperature": 0.1,
                "top_p": 1,
                "max_tokens": kwargs.get("max_tokens", 4096)
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"]

    if hashing_kv is not None:
        await hashing_kv.upsert(
            {args_hash: {"return": content, "model": model}}
        )
        await hashing_kv.index_done_callback()

    return content

async def deepseek_complete(model_name, prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    return await deepseek_complete_if_cache(
        model_name,
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        **kwargs
    )

@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def bge_m3_embedding(model_name: str, texts: list[str]) -> np.ndarray:
    # 使用硅基流动的BAAI/bge-m3嵌入模型
    import httpx
    
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://api.siliconflow.cn/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ.get('SILICONFLOW_API_KEY', 'sk-******')}",
                "Content-Type": "application/json"
            },
            json={
                "model": "BAAI/bge-m3",
                "input": texts,
                "encoding_format": "float"
            },
            timeout=60.0
        )
        response.raise_for_status()
        result = response.json()
        embeddings = [item["embedding"] for item in result["data"]]
        return np.array(embeddings)

# DeepSeek + BAAI/bge-m3 配置
deepseek_bge_config = LLMConfig(
    embedding_func_raw = bge_m3_embedding,
    embedding_model_name = "BAAI/bge-m3",
    embedding_dim = 1024,  # bge-m3的嵌入维度
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 16,
    query_better_than_threshold = 0.2,
    
    best_model_func_raw = deepseek_complete,
    best_model_name = "deepseek-chat",    
    best_model_max_token_size = 32768,
    best_model_max_async = 16,
    
    cheap_model_func_raw = deepseek_complete,
    cheap_model_name = "deepseek-chat",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 16
)

# Custom OpenAI-compatible API configuration
def create_custom_openai_config(base_url: str, api_key: str, model_name: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
    
    async def custom_embedding_wrapper(model_name: str, texts: list[str], **kwargs) -> np.ndarray:
        return await custom_openai_embedding(model_name, texts, base_url, api_key)
    
    async def custom_model_wrapper(model_name_inner, prompt, system_prompt=None, history_messages=[], **kwargs):
        return await custom_gpt_complete(
            model_name_inner, prompt, system_prompt=system_prompt, 
            history_messages=history_messages, base_url=base_url, api_key=api_key, **kwargs
        )
    
    return LLMConfig(
        embedding_func_raw = custom_embedding_wrapper,
        embedding_model_name = embedding_model,
        embedding_dim = 1536,  # text-embedding-3-small dimension
        embedding_max_token_size = 8192,
        embedding_batch_num = 32,
        embedding_func_max_async = 16,
        query_better_than_threshold = 0.2,
        
        best_model_func_raw = custom_model_wrapper,
        best_model_name = model_name,    
        best_model_max_token_size = 32768,
        best_model_max_async = 16,
        
        cheap_model_func_raw = custom_model_wrapper,
        cheap_model_name = model_name,
        cheap_model_max_token_size = 32768,
        cheap_model_max_async = 16
    )

# ==================================================================================
# == InternVL3_5-8B-HF (local Transformers) configuration (no Ollama dependency)
# ==================================================================================

# Globals to cache loaded resources
_internvl_model = None
_internvl_tokenizer = None
_internvl_processor = None

def _get_internvl_model_path() -> str:
    # Prefer explicit path, fallback to HF-style name under cache_dir
    return os.environ.get(
        "INTERNVL_MODEL_PATH",
        "/root/autodl-tmp/Model/OpenGVLab/InternVL3_5-8B-HF"
    )

def _ensure_internvl_loaded():
    global _internvl_model, _internvl_tokenizer, _internvl_processor
    if _internvl_model is not None and _internvl_tokenizer is not None:
        # 新增：返回三元组，供调用方解包
        return _internvl_model, _internvl_tokenizer, _internvl_processor
    model_path = _get_internvl_model_path()
    revision = os.environ.get("INTERNVL_REVISION", "") or None

    # Prefer ModelScope loader first for InternVL, then fallback to Transformers
    # 1) Try ModelScope
    # try:
    #     from modelscope.pipelines import pipeline
    #     from modelscope.utils.constant import Tasks
    #
    #     _internvl_model = pipeline(Tasks.multi_modal_text_generation, model=model_path)
    #     _internvl_tokenizer = None
    #     _internvl_processor = None
    #     return
    # except Exception:
    #     # Ignore and fallback to Transformers
    #     pass

    # 2) Fallback to Transformers
    try:
        from transformers import AutoModel, AutoTokenizer, AutoProcessor
        import torch
    except Exception as e:
        raise RuntimeError(
            "Transformers is required for InternVL3_5-8B-HF. Please `pip install -U transformers accelerate`"
        ) from e

    try:
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    except Exception:
        torch_dtype = None

    tf_kwargs = {
        "trust_remote_code": True,
        "device_map": "auto",
    }
    if revision:
        tf_kwargs["revision"] = revision
    if torch_dtype is not None:
        tf_kwargs["torch_dtype"] = torch_dtype  # type: ignore

    _internvl_model = AutoModel.from_pretrained(
        model_path,
        **tf_kwargs
    )
    _internvl_model.eval()

    tk_kwargs = {"trust_remote_code": True, "use_fast": False}
    if revision:
        tk_kwargs["revision"] = revision
    _internvl_tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        **tk_kwargs
    )

    try:
        _internvl_processor = AutoProcessor.from_pretrained(
            model_path,
            trust_remote_code=True,
            revision=revision if revision else None
        )
    except Exception:
        _internvl_processor = None

    # 新增：统一返回
    return _internvl_model, _internvl_tokenizer, _internvl_processor

async def _internvl_hf_complete_impl(
    prompt: str,
    images: list[str] | None = None,
    **kwargs,
):
    """
    InternVL-Chat-V1.5 completion implementation
    """
    if not prompt:
        return ""

    system_prompt = kwargs.get("system_prompt")
    images_base64: list[str] | None = kwargs.get("images_base64")
    max_new_tokens: int = int(kwargs.get("max_new_tokens", 512))
    temperature: float = float(kwargs.get("temperature", 0.1))
    top_p: float = float(kwargs.get("top_p", 1))

    # 1) 构造 PIL 图像列表（优先使用 images_base64，其次文件路径）
    def _b64_to_pil(b64s: list[str]) -> list[Image.Image]:
        out = []
        for s in b64s:
            try:
                if s.startswith("data:image"):
                    s = s.split(",", 1)[1]
                im = Image.open(BytesIO(base64.b64decode(s))).convert("RGB")
                out.append(im)
            except Exception:
                continue
        return out

    if images_base64:
        images_pil = _b64_to_pil(images_base64)
    else:
        images_pil = []
        if images:
            for p in images:
                try:
                    images_pil.append(Image.open(p).convert("RGB"))
                except Exception:
                    continue

    # 合并文本
    user_text = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"

    # 2) 同步执行体（供 run_in_executor 使用）
    def _run_sync_call(prompt_text: str, images_list: list[Image.Image]) -> str:
        model, tokenizer, proc = _ensure_internvl_loaded()
        final_prompt = ""
        try:
            if proc is None:
                raise ValueError("InternVL processor is not available.")

            # 无论如何都重新构建 prompt，确保格式正确
            n_img = len(images_list)

            if n_img > 0:
                # 清理原有的所有 <image> 标记和 Image-X: 前缀
                cleaned = re.sub(r'Image-\d+:\s*<image>\s*', '', prompt_text)
                cleaned = re.sub(r'<image>', '', cleaned).strip()
                
                # 按照 LMDeploy 的标准格式重新构建
                placeholder_parts = [f'Image-{i+1}: <image>' for i in range(n_img)]
                placeholder_str = '\n'.join(placeholder_parts)
                final_prompt = f"{placeholder_str}\n{cleaned}"
            else:
                # 没有图像时，清理所有图像相关标记
                final_prompt = re.sub(r'Image-\d+:\s*<image>\s*', '', prompt_text)
                final_prompt = re.sub(r'<image>', '', final_prompt).strip()

            # Debug输出，确认占位符与图片数一致
            print(f"[InternVL DEBUG] images_pil count: {n_img}, <image> count in prompt: {final_prompt.count('<image>')}")

            inputs = proc(
                text=final_prompt,
                images=images_list if n_img > 0 else None,
                return_tensors="pt"
            )
            # If caller provided a pre-built batch tensor, prefer it (saves reconstruction)
            if kwargs.get('images_tensors') is not None:
                try:
                    batch = kwargs.get('images_tensors')
                    # Ensure batch is on same device as model
                    try:
                        device = getattr(model, "device", None)
                        if device is None:
                            device = next(model.parameters()).device
                        batch = batch.to(device)
                    except Exception:
                        pass
                    # Replace image tensor in inputs if present
                    # Many processors return keys like 'pixel_values' or 'images'
                    for k in list(inputs.keys()):
                        if isinstance(inputs[k], torch.Tensor) and inputs[k].dim() == 4:
                            inputs[k] = batch
                except Exception:
                    # fall back to original inputs
                    pass
            else:
                # 尝试将张量迁移到模型设备（若单设备可用）
                try:
                    device = getattr(model, "device", None)
                    if device is None:
                        device = next(model.parameters()).device  # 可能是 cuda 或 cpu
                    moved = {}
                    for k, v in inputs.items():
                        moved[k] = v.to(device) if isinstance(v, torch.Tensor) else v
                    inputs = moved
                except Exception:
                    pass

            gen_kwargs = {
                "max_new_tokens": int(max_new_tokens),
                "do_sample": bool(temperature and temperature > 0.0),
                "temperature": float(temperature),
                "top_p": float(top_p),
            }

            with torch.no_grad():
                output_ids = model.generate(**inputs, **gen_kwargs)

            # 解码：若有 input_ids，则裁掉前缀
            try:
                input_len = inputs["input_ids"].shape[1]
                gen_only = output_ids[:, input_len:]
            except Exception:
                gen_only = output_ids

            try:
                text = tokenizer.decode(gen_only[0], skip_special_tokens=True)
            except Exception:
                # 备用解码
                text = tokenizer.batch_decode(gen_only, skip_special_tokens=True)[0]

            return text.strip()
        except Exception as e:
            # 根据用户要求，移除日志中冗长的prompt打印
            return f"InternVL generate error: {e}"

    loop = asyncio.get_running_loop()
    response = await loop.run_in_executor(
        None,
        _run_sync_call,
        user_text,
        images_pil
    )
    return response

async def internvl_hf_complete(
    model_name,
    prompt,
    system_prompt=None,
    history_messages=[],
    **kwargs
) -> str:
    # Accept images_base64 passed by caller
    images_base64 = kwargs.pop("images_base64", None)
    # ---- Fixed sampling defaults ----
    max_new = kwargs.pop("max_new_tokens", 512)
    temperature = kwargs.get("temperature", 0.1)
    top_p = kwargs.get("top_p", 1)

    # Delegate to implementation
    return await _internvl_hf_complete_impl(
        prompt=prompt,
        images=None,
        system_prompt=system_prompt,
        max_new_tokens=max_new,
        temperature=temperature,
        top_p=top_p,
        images_base64=images_base64,
    )

async def _dummy_embedding(_model_name: str, texts: list[str]) -> np.ndarray:
    # Minimal placeholder to satisfy interface; not used in current pipeline path
    arr = np.zeros((len(texts), 1536), dtype=float)
    return arr

internvl_hf_config = LLMConfig(
    embedding_func_raw = _dummy_embedding,
    embedding_model_name = "dummy-embeddings",
    embedding_dim = 1536,
    embedding_max_token_size = 8192,
    embedding_batch_num = 32,
    embedding_func_max_async = 8,
    query_better_than_threshold = 0.2,

    best_model_func_raw = internvl_hf_complete,
    best_model_name = "OpenGVLab/InternVL3_5-8B-HF",
    best_model_max_token_size = 32768,
    best_model_max_async = 2,

    cheap_model_func_raw = internvl_hf_complete,
    cheap_model_name = "OpenGVLab/InternVL3_5-8B-HF",
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 2
)


# Generic local HF wrapper config (routes to vllm/transformers/internvl depending on short name)
local_hf_generic_config = LLMConfig(
    embedding_func_raw = hf_local_embedding if 'hf_local_embedding' in globals() else _dummy_embedding,
    embedding_model_name = DEFAULT_OLLAMA_EMBED_MODEL,
    embedding_dim = 768,
    embedding_max_token_size = 8192,
    embedding_batch_num = 8,
    embedding_func_max_async = 4,
    query_better_than_threshold = 0.2,

    best_model_func_raw = lambda model_name, prompt, **kw: local_complete_router(model_name.split(":",1)[0].lower(), prompt, **kw),
    best_model_name = DEFAULT_OLLAMA_CHAT_MODEL,
    best_model_max_token_size = 32768,
    best_model_max_async = 2,

    cheap_model_func_raw = lambda model_name, prompt, **kw: local_complete_router(model_name.split(":",1)[0].lower(), prompt, **kw),
    cheap_model_name = DEFAULT_OLLAMA_CHAT_MODEL,
    cheap_model_max_token_size = 32768,
    cheap_model_max_async = 2,
)

# ==================================================================================
# == Specific model for Refinement Evaluation
# ==================================================================================
_refiner_client = None

def get_deepseek_r1_refiner_client():
    global _refiner_client
    if _refiner_client is None:
        if not _OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI client is not available for Refiner. Please install 'openai'.")
        _refiner_client = AsyncOpenAI(
            base_url="https://api2.aigcbest.top/v1",
            api_key="sk-X5tCPutvJTOsTpSHnl2bz3IF0EJkFjK22HekxMVUIQQjhNEm"
        )
    return _refiner_client

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def deepseek_r1_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call DeepSeek-R1 for refinement evaluation.
    It uses a specific client instance and does not interfere with the main LLM config.
    """
    client = get_deepseek_r1_refiner_client()
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.chat.completions.create(
        model="deepseek-ai/DeepSeek-R1",
        messages=messages,
        # 统一固定采样参数
        temperature=0.1,
        top_p=1,
        max_tokens=512,
        **kwargs
    )
    # DeepSeek-R1 may return content=None and put text in reasoning_content
    choice = response.choices[0].message
    text = getattr(choice, "content", None)
    if not text:
        text = getattr(choice, "reasoning_content", None)
    return text or ""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=6),
    retry=retry_if_exception_type((RateLimitError, APIConnectionError)),
)
async def ollama_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call an Ollama model for refinement evaluation.
    """
    # For multi-modal models like llava, the normal chat endpoint might hang on text-only prompts.
    # Delegate to a simpler completion function.
    if "llava" in model_name:
        # 仍走通用通道，但已在 ollama_complete_if_cache 中统一固定 temperature/top_p
        return await ollama_complete(model_name, prompt, **kwargs)
    # If short name maps to local HF, use local wrapper
    short = (model_name or "").split(":")[0].lower()
    if short in MODEL_NAME_TO_LOCAL_PATH:
        return await hf_local_text_complete(model_short_name=short, prompt=prompt, **kwargs)

    client = get_ollama_async_client_instance()
    messages = [{"role": "user", "content": prompt}]
    
    response = await client.chat(
        model=model_name,
        messages=messages,
        options={
            # 统一固定采样参数
            "temperature": 0.1,
            "top_p": 1,
        },
        keep_alive=-1  # Keep the model loaded in memory
    )
    return response['message']['content']


async def internvl_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """
    A dedicated, non-cached function to call the local InternVL (HF) model
    for refinement evaluation and keyword generation when Ollama is not used.
    """
    # Delegate to the InternVL HF completion with minimal settings
    return await internvl_hf_complete(
        model_name=model_name,
        prompt=prompt,
        system_prompt=kwargs.get("system_prompt")
    )


# Backwards-compatible neutral wrappers (avoid leaving only 'ollama_' internal names)
async def external_llm_refiner_func(model_name: str, prompt: str, **kwargs) -> str:
    """Neutral wrapper that routes to the legacy ollama refiner implementation or local HF.
    Keeps a clear, non-Ollama name in call sites while preserving behaviour.
    """
    return await ollama_refiner_func(model_name=model_name, prompt=prompt, **kwargs)


def get_default_external_llm_chat_model() -> str:
    """Return default chat model (alias for get_default_ollama_chat_model for compatibility)."""
    return get_default_ollama_chat_model()

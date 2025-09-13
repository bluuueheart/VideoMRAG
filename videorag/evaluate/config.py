import os

try:
    # Suppress excessive warnings from transformers
    from transformers.utils.logging import set_verbosity_error
    set_verbosity_error()
    # Use the same default model resolver as generation
    from .._llm import get_default_ollama_chat_model
except Exception:  # pragma: no cover
    def get_default_ollama_chat_model() -> str:  # type: ignore
        return os.environ.get("OLLAMA_CHAT_MODEL", "qwen2.5vl:7b")


# ---------------- Config ----------------
INPUT_BASE_DIR = "/root/autodl-tmp/10test"

def _infer_model_tag(model_name: str) -> str:
    name = (model_name or "").lower()
    if "qwen" in name:
        return "qwen"
    if "llama" in name:
        return "llama"
    if "gemma" in name:
        return "gemma"
    return (name.split(":", 1)[0] or "misc").replace("/", "_")

_active_chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "").strip() or get_default_ollama_chat_model()
OUTPUT_BASE_DIR = f"/root/autodl-tmp/Result/{_infer_model_tag(_active_chat_model)}"
# New: scan all result jsons under the Result root
OUTPUT_ROOT_DIR = "/root/autodl-tmp/Result"
# Ground-truth data root (for question-based lookup)
DATA_ROOT_DIR = "/root/autodl-tmp/Data"

# Default local caches to /root/autodl-tmp
DEFAULT_OLLAMA_DIR = "/root/autodl-tmp/ollama"
DEFAULT_HF_HOME = "/root/autodl-tmp/huggingface"
os.environ.setdefault("OLLAMA_MODELS", DEFAULT_OLLAMA_DIR)
os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(DEFAULT_HF_HOME, "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(DEFAULT_HF_HOME, "transformers"))
os.makedirs(os.environ.get("OLLAMA_MODELS", DEFAULT_OLLAMA_DIR), exist_ok=True)
os.makedirs(os.environ.get("HF_HOME", DEFAULT_HF_HOME), exist_ok=True)
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# 显式评估模型可由环境变量覆盖
EVAL_LLM_MODEL = os.environ.get("EVAL_LLM_MODEL", "qwen3:8b")
# Always use a fast and reliable model for evaluation (kept name for downstream compatibility)
OLLAMA_MODEL = EVAL_LLM_MODEL
# Prefer locally-downloaded ModelScope roberta-base if available
MODELSCOPE_BASE_DIR = os.environ.get("MODELSCOPE_CACHE_DIR", "/root/autodl-tmp/Model")
MODELSCOPE_ROBERTA_DIR = os.environ.get(
    "MODELSCOPE_ROBERTA_DIR",
    os.path.join(MODELSCOPE_BASE_DIR, "AI-ModelScope", "roberta-base"),
)

def _resolve_default_bertscore_model() -> str:
    """
    自动解析 BERTScore 模型。
    优先级:
    1. 环境变量 `BERTSCORE_MODEL` (最高)
    2. 本地 sentence-transformers MiniLM (优先：轻量且 safetensors，不触发 torch>=2.6 限制)
    3. 本地 DeBERTa
    4. 其他可能的本地模型 (预留 roberta-base)
    5. 默认回退在线 DeBERTa
    """
    env_val = (os.environ.get("BERTSCORE_MODEL", "").strip())
    if env_val:
        return env_val

    # 优先轻量 MiniLM sentence-transformers (ModelScope 默认路径)
    minilm_path = os.path.join(MODELSCOPE_BASE_DIR, "sentence-transformers", "all-MiniLM-L6-v2")
    if os.path.isdir(minilm_path) and os.path.exists(os.path.join(minilm_path, "config.json")):
        print(f"[BERTScore Info] Found local MiniLM model: {minilm_path}")
        return minilm_path

    # 回退到在线模型
    print("[BERTScore Info] No local model found, falling back to online 'microsoft/deberta-v3-base'.")
    return "microsoft/deberta-v3-base"

# 强制使用指定本地 SentenceTransformer 模型（硬编码）
FORCED_ST_MODEL = "/root/autodl-tmp/Model/sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_BERTSCORE_MODEL = FORCED_ST_MODEL
DEFAULT_BERTSCORE_BATCH = int(os.environ.get("BERTSCORE_BATCH", "32"))

# 硬编码默认导出，优先使用本地 sentence-transformers 模型
os.environ.setdefault("BERTSCORE_MODEL", FORCED_ST_MODEL)

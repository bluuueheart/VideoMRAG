# videorag/_config.py
# This file holds shared, static configuration variables to prevent circular imports.
import os

# Root prefix configuration: centralize project root overrides.
# Set environment variable `ROOT_PREFIX` to one of:
#   /mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02
#   /home/hadoop-aipnlp/dolphins_hdd_hadoop-aipnlp/KAI/gaojinpeng02
# If unset, detection will prefer mounted /mnt then local /home.
_CACHED_ROOT_PREFIX = None
# Optional manual override: you may directly edit this variable in-place (or set env ROOT_PREFIX)
# 服务器目录：/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/
# 本地目录：/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/
ROOT_PREFIX_OVERRIDE = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/'

def get_root_prefix() -> str:
    global _CACHED_ROOT_PREFIX
    if _CACHED_ROOT_PREFIX:
        return _CACHED_ROOT_PREFIX

    # 1) explicit in-memory override (useful for quickly editing the config file)
    if ROOT_PREFIX_OVERRIDE:
        _CACHED_ROOT_PREFIX = ROOT_PREFIX_OVERRIDE.rstrip('/')
        return _CACHED_ROOT_PREFIX

    # 2) environment variable overrides
    env = os.environ.get('ROOT_PREFIX') or os.environ.get('MODEL_ROOT')
    if env:
        _CACHED_ROOT_PREFIX = env.rstrip('/')
        return _CACHED_ROOT_PREFIX

    # 3) auto-detection (prefer mounted /mnt then local /home)
    mnt_candidate = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/'
    home_candidate = '/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/'
    if os.path.isdir(mnt_candidate):
        _CACHED_ROOT_PREFIX = mnt_candidate
    elif os.path.isdir(home_candidate):
        _CACHED_ROOT_PREFIX = home_candidate
    else:
        _CACHED_ROOT_PREFIX = home_candidate
    return _CACHED_ROOT_PREFIX

# Expose ROOT_PREFIX constant and helper for backwards compatibility
ROOT_PREFIX = get_root_prefix()

def get_model_root() -> str:
    """Model root directory derived from chosen ROOT_PREFIX.
    Returns e.g. '<ROOT_PREFIX>/00_opensource_models'.
    """
    return os.path.join(ROOT_PREFIX, '00_opensource_models')

# Backwards-compatible constant (evaluated on import)
MODEL_ROOT = get_model_root()

# Build common model paths relative to MODEL_ROOT (these can be used as defaults)
MINICPM_MODEL_PATH = os.path.join(MODEL_ROOT, 'huggingface.co', 'openbmb', 'MiniCPM-V-4_5')
# YOLO-World detector model path
YOLOWORLD_MODEL_PATH = os.path.join(MODEL_ROOT, 'yolov8m-worldv2.pt')

# Derived defaults based on chosen ROOT_PREFIX (editable via ROOT_PREFIX_OVERRIDE or env)
# These provide a single place to change shared deployment paths.
OUTPUT_BASE_DIR_DEFAULT = os.path.join(ROOT_PREFIX, 'lx', 'Result')
DATA_ROOT_DIR_DEFAULT = os.path.join(ROOT_PREFIX, 'lx', 'Benchmark')
DEFAULT_EXTERNAL_MODELS_DIR = os.path.join(ROOT_PREFIX, 'lx', 'llm_models')
DEFAULT_HF_HOME = os.path.join(ROOT_PREFIX, 'lx', 'huggingface')
# Default faster-whisper model path under the model root
FASTER_WHISPER_DEFAULT = os.path.join(MODEL_ROOT, 'huggingface.co', 'deepdml', 'faster-distil-whisper-large-v3.5')

# --- Frame refinement / de-dup config defaults ---
# 感知哈希汉明距离阈值（越小越严格）
DEDUP_PHASH_THRESHOLD_DEFAULT = 5
# 是否输出调试统计
DEDUP_DEBUG = False
# 扩展帧数映射：键 = 分数(0~10)，值 = 目标帧数（示例：分数高 → 需要更多帧；你可按需再调）
FRAME_COUNT_MAPPING_EXTENDED = {
    0: 40,
    1: 36,
    2: 32,
    3: 28,
    4: 24,
    5: 20,
}

# ------------------------------------------------------------------
# New refinement / runtime control defaults (centralized tuning)
# 可在此集中调参，后续代码可优先读取这些常量，再允许环境变量覆盖。
# 若需要环境变量覆盖策略，可在使用处：
#   timeout = float(os.environ.get("REFINER_TIMEOUT_SECONDS", REFINER_TIMEOUT_SECONDS_DEFAULT))
# ------------------------------------------------------------------

# 细化阶段上下文评估（计划生成）超时（秒）
REFINER_TIMEOUT_SECONDS_DEFAULT = 60
# 超时后的回退策略: "final" (直接跳过 refine) 或 "refine" (全量 refine)
REFINER_TIMEOUT_FALLBACK_DEFAULT = "final"
# 并发最大并行评估任务
REFINER_MAX_PARALLEL_DEFAULT = 1
# 关键词（DET 用）生成超时（秒）
REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT = 30

# OCR / DET 抽帧默认数
REFINE_OCR_FRAMES_DEFAULT = 6
REFINE_DET_FRAMES_DEFAULT = 8

# 差异增量摘要（diff summary）相关
REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT = 160
REFINE_DIFF_MAX_FRAMES_DEFAULT = 12
REFINE_DIFF_DECAY_WINDOW_DEFAULT = 3

# Keyframe -> token 估算系数 (approx tokens per raw byte, 用于日志统计)
KEYFRAME_TOKENS_PER_BYTE_DEFAULT = 1.0 / 3.0

# 可选：最大目标帧数上限保护（避免异常配置导致过度抽帧）
REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT = 40

__all__ = [
    "MINICPM_MODEL_PATH",
    "MODEL_ROOT",
    "ROOT_PREFIX",
    "OUTPUT_BASE_DIR_DEFAULT",
    "DATA_ROOT_DIR_DEFAULT",
    "DEFAULT_EXTERNAL_MODELS_DIR",
    "DEFAULT_HF_HOME",
    "FASTER_WHISPER_DEFAULT",
    "DEDUP_PHASH_THRESHOLD_DEFAULT",
    "DEDUP_DEBUG",
    "FRAME_COUNT_MAPPING_EXTENDED",
    # New exports
    "REFINER_TIMEOUT_SECONDS_DEFAULT",
    "REFINER_TIMEOUT_FALLBACK_DEFAULT",
    "REFINER_MAX_PARALLEL_DEFAULT",
    "REFINER_KEYWORD_TIMEOUT_SECONDS_DEFAULT",
    "REFINE_OCR_FRAMES_DEFAULT",
    "REFINE_DET_FRAMES_DEFAULT",
    "REFINE_DIFF_SUMMARY_MAX_CHARS_DEFAULT",
    "REFINE_DIFF_MAX_FRAMES_DEFAULT",
    "REFINE_DIFF_DECAY_WINDOW_DEFAULT",
    "KEYFRAME_TOKENS_PER_BYTE_DEFAULT",
    "REFINE_GLOBAL_MAX_FRAMES_PER_30S_DEFAULT",
]

import os
import logging
from tqdm import tqdm

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None


def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    """Transcribe segments using faster-whisper.

    The loader is defensive: it will try to initialize WhisperModel with
    the `local_files_only` flag when supported, and prints clear guidance
    when model files are not present or online downloads are disabled.
    Use environment `FASTER_WHISPER_DIR` or `ASR_MODEL_PATH` to point to
    a local snapshot. To allow automatic downloads from the Hub, set
    `ALLOW_HF_DOWNLOADS=1` in the environment.
    """
    # Prefer environment overrides: FASTER_WHISPER_DIR or ASR_MODEL_PATH
    # Then prefer configured default from videorag._config (which respects ROOT_PREFIX_OVERRIDE)
    # Do NOT fallback to repo-relative or /workdir paths. This environment is offline-only
    # so we force local model usage and disallow online downloads.
    try:
        from videorag._config import FASTER_WHISPER_DEFAULT, MODEL_ROOT
    except Exception:
        FASTER_WHISPER_DEFAULT = None
        MODEL_ROOT = None

    env_model = os.environ.get("FASTER_WHISPER_DIR") or os.environ.get("ASR_MODEL_PATH")
    if env_model:
        model_dir = env_model
    elif FASTER_WHISPER_DEFAULT:
        model_dir = FASTER_WHISPER_DEFAULT
    elif MODEL_ROOT:
        # build the canonical path under MODEL_ROOT
        model_dir = os.path.join(MODEL_ROOT, 'huggingface.co', 'deepdml', 'faster-distil-whisper-large-v3.5')
    else:
        model_dir = None

    if WhisperModel is None:
        raise RuntimeError("faster_whisper.WhisperModel is not importable. Please install 'faster-whisper'.")

    # Offline-only: always disable online downloads and require local model files
    allow_online = False

    # Validate model_dir is set and points to an existing directory
    if not model_dir:
        print("[ASR] No local model path resolved. Please set FASTER_WHISPER_DIR or ASR_MODEL_PATH.")
        raise RuntimeError("No faster-whisper model directory configured for offline use.")

    if not os.path.isdir(model_dir):
        print(f"[ASR] Resolved model_dir does not exist: {model_dir}")
        if MODEL_ROOT:
            suggested = os.path.join(MODEL_ROOT, 'huggingface.co', 'deepdml', 'faster-distil-whisper-large-v3.5')
            print(f"Suggested default path based on ROOT_PREFIX: {suggested}")
        raise RuntimeError(f"faster-whisper model directory not found: {model_dir}")

    # Debug: print resolved model_dir for easier diagnosis
    try:
        print(f"[ASR] Resolved faster-whisper model_dir: {model_dir}")
    except Exception:
        pass

    # Choose device heuristically (prefer GPU when available)
    device = "cpu"
    try:
        if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
            import torch

            if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                device = "cuda"
                print(f"[ASR] Detected CUDA devices. Preferring device='{device}' for ASR.")
    except Exception:
        # keep CPU if any import/device check fails
        device = "cpu"

    compute_type = "int8"

    model = None
    # Try to pass local_files_only when available; fall back if API differs
    try:
        try:
            model = WhisperModel(model_dir, device=device, compute_type=compute_type, local_files_only=True)
        except TypeError:
            # some faster-whisper releases may not support the kwarg
            model = WhisperModel(model_dir, device=device, compute_type=compute_type)
        model.logger.setLevel(logging.WARNING)
    except Exception as e:
        print(f"[ASR] Failed to load faster-whisper model from '{model_dir}': {e}")
        print("[ASR] This environment requires local models and online downloads are disabled.")
        print("[ASR] Please ensure the faster-whisper model is present locally and set one of:")
        print("  - environment variable FASTER_WHISPER_DIR or ASR_MODEL_PATH pointing to the model folder")
        if MODEL_ROOT:
            suggested = os.path.join(MODEL_ROOT, 'huggingface.co', 'deepdml', 'faster-distil-whisper-large-v3.5')
            print(f"Suggested default path based on ROOT_PREFIX: {suggested}")
        raise

    cache_path = os.path.join(working_dir, "_cache", video_name)

    transcripts = {}
    for index in tqdm(segment_index2name, desc=f"Speech Recognition {video_name}"):
        segment_name = segment_index2name[index]
        audio_file = os.path.join(cache_path, f"{segment_name}.{audio_output_format}")
        segments, info = model.transcribe(audio_file)
        result = ""
        for segment in segments:
            result += "[%.2fs -> %.2fs] %s\n" % (segment.start, segment.end, segment.text)
        transcripts[index] = result

    return transcripts
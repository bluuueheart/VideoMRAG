import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
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
    # Finally fallback to repo-relative path
    try:
        from videorag._config import FASTER_WHISPER_DEFAULT
    except Exception:
        FASTER_WHISPER_DEFAULT = None

    model_dir = (
        os.environ.get("FASTER_WHISPER_DIR")
        or os.environ.get("ASR_MODEL_PATH")
        or (FASTER_WHISPER_DEFAULT if FASTER_WHISPER_DEFAULT else None)
        or os.path.join(os.path.dirname(__file__), "..", "..", "faster-distil-whisper-large-v3")
    )

    if WhisperModel is None:
        raise RuntimeError("faster_whisper.WhisperModel is not importable. Please install 'faster-whisper'.")

    # Determine whether online downloads are allowed (opt-in)
    allow_online = str(os.environ.get("ALLOW_HF_DOWNLOADS", "0")).lower() in {"1", "true", "yes"}

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
            model = WhisperModel(model_dir, device=device, compute_type=compute_type, local_files_only=(not allow_online))
        except TypeError:
            # some faster-whisper releases may not support the kwarg
            model = WhisperModel(model_dir, device=device, compute_type=compute_type)
        model.logger.setLevel(logging.WARNING)
    except Exception as e:
        print(f"[ASR] Failed to load faster-whisper model from '{model_dir}': {e}")
        if not allow_online:
            print("[ASR] Online downloads are disabled. To allow automatic downloads set environment: ALLOW_HF_DOWNLOADS=1")
            print("[ASR] Or pre-download the model to a local folder and set FASTER_WHISPER_DIR or ASR_MODEL_PATH to that folder.")
        else:
            print("[ASR] Model loading failed even though online downloads are allowed. Check network or model name/path.")
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
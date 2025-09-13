import os
import torch
import logging
from tqdm import tqdm
from faster_whisper import WhisperModel
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

def speech_to_text(video_name, working_dir, segment_index2name, audio_output_format):
    # Prefer environment overrides: FASTER_WHISPER_DIR or ASR_MODEL_PATH, fallback to repo-relative path
    model_dir = os.environ.get("FASTER_WHISPER_DIR") or os.environ.get("ASR_MODEL_PATH") or os.path.join(os.path.dirname(__file__), "..", "..", "faster-distil-whisper-large-v3")
    model = WhisperModel(model_dir)
    model.logger.setLevel(logging.WARNING)
    
    cache_path = os.path.join(working_dir, '_cache', video_name)
    
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
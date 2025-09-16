import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import the model path from the new central config file
from .._config import MINICPM_MODEL_PATH

torch.set_float32_matmul_precision('medium')

# Prefer fast image processor in Transformers
os.environ.setdefault("TRANSFORMERS_IMAGE_PROCESSING_USE_FAST", "1")

def encode_video(video, frame_times):
    frames = []
    duration = getattr(video, "duration", None)
    if duration is None:
        duration = 0.0
    max_t = max(0.0, float(duration) - 1e-3)

    # Clamp sampling times to valid range to avoid boundary decoding issues
    safe_times = frame_times
    try:
        if len(frame_times) > 0:
            safe_times = np.clip(frame_times, 0.0, max_t)
    except Exception:
        # If frame_times is not a numpy array, fall back to per-item clamp
        safe_times = [min(max(float(t), 0.0), max_t) for t in frame_times]

    for t in safe_times:
        t0 = float(t)
        try:
            frame = video.get_frame(t0)
        except Exception:
            # Nudge slightly earlier to stay within decodable range without reducing frame count
            t1 = max(0.0, min(max_t, t0 - 1e-2))
            frame = video.get_frame(t1)
        frames.append(frame)

    # Convert to RGB PIL images at target resolution
    frames = [Image.fromarray(np.asarray(v).astype('uint8')).convert('RGB').resize((1280, 720)) for v in frames]
    return frames
    
def _load_minicpm():
    """Prefer GPU (RTX 3080 Ti single card) and fall back to CPU only on OOM.
    - Primary: float16 + device_map='auto' (uses cuda:0 if available)
    - Fallback (ONLY on CUDA OOM): float32 + CPU
    """
    model_path = os.environ.get("MINICPM_MODEL_PATH", MINICPM_MODEL_PATH)
    try:
        try:
            model = AutoModel.from_pretrained(
                model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map='auto',
                low_cpu_mem_usage=True,
            )
        except Exception as e_inner:
            msg = str(e_inner) or ""
            if "meta tensor" in msg or "Cannot copy out of meta tensor" in msg:
                # Fallback to CPU-only load
                model = AutoModel.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    device_map={'': 'cpu'}
                )
            else:
                raise
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        model.eval()
        return model, tokenizer
    except Exception as e:
        # Only fall back to CPU when it's an OOM error
        is_oom = False
        try:
            import torch
            is_oom = isinstance(e, torch.cuda.OutOfMemoryError)
        except Exception:
            pass
        text = str(e).lower()
        if (not is_oom) and ("out of memory" in text or "cuda oom" in text):
            is_oom = True
        if not is_oom:
            # Not an OOM -> respect policy: do not auto-CPU fallback
            raise
        # OOM fallback: CPU float32
            try:
                # Attempt accelerate dispatch across GPUs rather than falling back to CPU
                from accelerate import init_empty_weights, load_checkpoint_and_dispatch
                with init_empty_weights():
                    model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
                model = load_checkpoint_and_dispatch(model, model_path, device_map="auto")
                tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token
                model.eval()
                return model, tokenizer
            except Exception:
                # Surface the original OOM for clarity
                raise e
    

def segment_caption(video_name, video_path, segment_index2name, transcripts, segment_times_info, caption_result, error_queue):
    try:
        model, tokenizer = _load_minicpm()
        
        with VideoFileClip(video_path) as video:
            for index in tqdm(segment_index2name, desc=f"Captioning Video {video_name}"):
                frame_times = segment_times_info[index]["frame_times"]
                video_frames = encode_video(video, frame_times)
                segment_transcript = transcripts[index]
                query = f"The transcript of the current video:\n{segment_transcript}.\nNow provide a description (caption) of the video in English."
                # Pass images via dedicated parameter to avoid empty image batch parsing
                msgs = [{'role': 'user', 'content': query}]
                params = {
                    "use_image_id": False,
                    "max_slice_nums": 2,
                    "temperature": 0.2,
                    "do_sample": False,
                    "max_new_tokens": 256,
                }
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                    seg_cap = model.chat(
                        image=video_frames,
                        msgs=msgs,
                        tokenizer=tokenizer,
                        **params
                    )
                caption_result[index] = seg_cap.replace("\n", "").replace("<|endoftext|>", "")
                torch.cuda.empty_cache()
    except Exception as e:
        error_queue.put(f"Error in segment_caption:\n {str(e)}")
        raise RuntimeError


def merge_segment_information(segment_index2name, segment_times_info, transcripts, captions):
    inserting_segments = {}
    for index in segment_index2name:
        inserting_segments[index] = {"content": None, "time": None}
        segment_name = segment_index2name[index]
        inserting_segments[index]["time"] = '-'.join(segment_name.split('-')[-2:])
        inserting_segments[index]["content"] = f"Caption:\n{captions[index]}\nTranscript:\n{transcripts[index]}\n\n"
        inserting_segments[index]["transcript"] = transcripts[index]
        inserting_segments[index]["frame_times"] = segment_times_info[index]["frame_times"].tolist()
    return inserting_segments
        

def retrieved_segment_caption(caption_model, caption_tokenizer, refine_knowledge, retrieved_segments, video_path_db, video_segments, num_sampled_frames, extra_text_map=None):
    # caption_model / caption_tokenizer are provided by VideoRAG.load_caption_model()
    
    caption_result = {}
    for this_segment in tqdm(retrieved_segments, desc='Captioning Segments for Given Query'):
        video_name = '_'.join(this_segment.split('_')[:-1])
        index = this_segment.split('_')[-1]
        video_path = video_path_db._data[video_name]
        timestamp = video_segments._data[video_name][index]["time"].split('-')
        start, end = eval(timestamp[0]), eval(timestamp[1])
        video = VideoFileClip(video_path)
        # Guard against degenerate intervals by slightly nudging end and clamping later
        if end <= start:
            end = min(getattr(video, "duration", end) or end, start + 1e-3)
        frame_times = np.linspace(start, end, num_sampled_frames, endpoint=False)
        video_frames = encode_video(video, frame_times)
        if not isinstance(video_frames, list) or len(video_frames) == 0:
            raise RuntimeError(f"Empty frames for segment {this_segment} with time [{start}, {end}] from {video_path}")
        segment_transcript = video_segments._data[video_name][index]["transcript"]
        extra_text = ""
        if extra_text_map is not None and this_segment in extra_text_map and extra_text_map[this_segment].strip():
            extra_text = f"\nAdditional Visual Text/Objects:\n{extra_text_map[this_segment]}\n"
        # remove trailing stray quote and set stable generation params
        query = (
            f"The transcript of the current video:\n{segment_transcript}.{extra_text}\n"
            f"Now provide a very detailed description (caption) of the video in English and extract relevant information about: {refine_knowledge}."
        )
        # Pass images via dedicated parameter to avoid empty image batch parsing
        msgs = [{'role': 'user', 'content': query}]
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,
            # remove unsupported sampling flags for this model interface
            "max_new_tokens": 256,
        }
        params = {
            "use_image_id": False,
            "max_slice_nums": 2,
            # remove unsupported sampling flags for this model interface
            "max_new_tokens": 256,
        }
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            segment_caption = caption_model.chat(
                image=video_frames,
                msgs=msgs,
                tokenizer=caption_tokenizer,
                **params
            )
        this_caption = segment_caption.replace("\n", "").replace("<|endoftext|>", "")
        caption_result[this_segment] = f"Caption:\n{this_caption}\nTranscript:\n{segment_transcript}\n\n"
        torch.cuda.empty_cache()
    
    return caption_result
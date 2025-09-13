import os
import numpy as np
import torch
from typing import List, Dict, Any, Optional
from PIL import Image
from tqdm import tqdm
from moviepy.video.io.VideoFileClip import VideoFileClip

# EasyOCR
try:
    import easyocr  # type: ignore
    _HAS_EASYOCR = True
except Exception:
    _HAS_EASYOCR = False

# OWL-ViT (open-vocabulary detection)
try:
    # Correct class names in transformers: OwlViT*
    from transformers import OwlViTProcessor as _OwlProcessor, OwlViTForObjectDetection as _OwlModel  # type: ignore
    _HAS_OWLVIT = True
except Exception:
    try:
        # Backward-compat alias just in case some environments expose OWLVit*
        from transformers import OWLVitProcessor as _OwlProcessor, OWLVitForObjectDetection as _OwlModel  # type: ignore
        _HAS_OWLVIT = True
    except Exception:
        _HAS_OWLVIT = False


def _sample_frames(
    video_path: str, 
    start: float, 
    end: float, 
    num_frames: int, 
    exclude_timestamps: Optional[List[float]] = None
) -> List[Image.Image]:
    if num_frames <= 0:
        return []
    
    # Generate potential frame timestamps
    frame_times = np.linspace(start, end, num_frames, endpoint=False)
    
    # If there are timestamps to exclude, filter them out
    if exclude_timestamps and len(exclude_timestamps) > 0:
        # For each new timestamp, check if it's too close to any excluded timestamp
        # A simple threshold of 0.1s should be sufficient to avoid duplicates
        final_times = []
        for t_new in frame_times:
            is_too_close = False
            for t_old in exclude_timestamps:
                if abs(t_new - t_old) < 0.1:
                    is_too_close = True
                    break
            if not is_too_close:
                final_times.append(t_new)
        frame_times = np.array(final_times)

    if len(frame_times) == 0:
        return []

    with VideoFileClip(video_path) as video:
        frames = [video.get_frame(t) for t in frame_times]
        
    pil_frames = [Image.fromarray(f.astype("uint8")) for f in frames]
    return pil_frames


def extract_ocr_text_for_segments(
    segments: List[str],
    video_path_db,
    video_segments,
    num_frames: int,
    languages: List[str] | None = None,
    pre_extracted_frames: Dict[str, List] | None = None,
) -> Dict[str, str]:
    """
    Extract OCR text for each segment by sampling frames and running EasyOCR.
    Returns a map: segment_id -> concatenated_text
    """
    if not _HAS_EASYOCR:
        print("[Refine-OCR] Skipped: easyocr library is not installed.")
        return {}
    
    try:
        if languages is None:
            # English + Simplified Chinese by default
            languages = ["en", "ch_sim"]
        # Prefer GPU if available, but gracefully fall back to CPU on any CUDA/cuDNN/runtime errors
        use_gpu = torch.cuda.is_available() and os.environ.get("EASYOCR_FORCE_CPU", "0").strip() not in {"1", "true", "True"}
        try:
            reader = easyocr.Reader(languages, gpu=use_gpu)
        except Exception as gpu_err:
            # Common causes: cuDNN version mismatch, driver/runtime conflicts, missing CUDA libraries, etc.
            print(f"[Refine-OCR] GPU init failed ({gpu_err}). Falling back to CPU...")
            # Explicitly retry with CPU to avoid CUDA dependencies
            reader = easyocr.Reader(languages, gpu=False)
    except Exception as e:
        print(f"[Refine-OCR] Failed to initialize EasyOCR reader. Error: {e}")
        # This can happen due to model download issues or CUDA/driver incompatibilities.
        return {}

    seg2text: Dict[str, str] = {}
    total_frames = 0
    total_text_tokens = 0
    try:
        for s_id in tqdm(segments, desc="Refine Pass: OCR"):
            try:
                # Prefer pre-extracted frames if provided
                frames: List[Image.Image] = []
                existing_frames: List[Image.Image] = []
                existing_timestamps: List[float] = []
                
                if pre_extracted_frames is not None and s_id in pre_extracted_frames:
                    raw_list = pre_extracted_frames.get(s_id) or []
                    # raw_list may be [(path, ts), ...] or [path, ...]
                    for item in raw_list:
                        try:
                            path, ts = (item[0], item[1]) if isinstance(item, (list, tuple)) and len(item) > 1 else (item, -1.0)
                            existing_frames.append(Image.open(str(path)).convert("RGB"))
                            if ts >= 0:
                                existing_timestamps.append(float(ts))
                        except Exception:
                            continue
                
                # Sample additional frames, excluding existing ones
                video_name, start, end = _get_segment_time(video_segments, s_id)
                video_path = video_path_db._data[video_name]
                
                # Calculate how many *new* frames to sample
                num_new_frames = max(0, num_frames - len(existing_frames))
                
                new_frames = _sample_frames(
                    video_path, 
                    start, 
                    end, 
                    num_new_frames, 
                    exclude_timestamps=existing_timestamps
                )
                
                # Combine existing and new frames
                frames = existing_frames + new_frames
                
                if not frames:
                    continue

                total_frames += len(frames)
                ocr_texts: List[str] = []
                for img in frames:
                    # easyocr accepts numpy arrays (RGB)
                    result = reader.readtext(np.array(img))
                    if not result:
                        continue
                    for _, text, conf in result:
                        if isinstance(text, str) and text.strip():
                            ocr_texts.append(text.strip())
                if len(ocr_texts):
                    # unique & join
                    uniq = []
                    seen = set()
                    for t in ocr_texts:
                        if t not in seen:
                            uniq.append(t)
                            seen.add(t)
                    seg2text[s_id] = " ".join(uniq)
                    total_text_tokens += len(seg2text[s_id].split())
            except Exception as seg_err:
                print(f"[Refine-OCR] Error processing segment {s_id}: {seg_err}")
                continue # Continue with the next segment
        print(f"[Refine-OCR] segments={len(segments)} frames={total_frames} tokens={total_text_tokens}")
    except Exception as e:
        print(f"[Refine-OCR] A critical error occurred during the OCR process: {e}")
        # Return any data processed so far
        return seg2text
    return seg2text

# Helper: 获取片段时间窗口 (video_name, start, end)
def _get_segment_time(video_segments, segment_id: str):
    """Safely fetch (video_name, start, end) for a segment id.
    Expected structure: video_segments[segment_id] = { 'video': name, 'start': float, 'end': float }
    Fallbacks to (segment_id, 0, 0) if missing.
    """
    try:
        meta = video_segments.get(segment_id) if isinstance(video_segments, dict) else None
        if not meta:
            return segment_id, 0.0, 0.0
        video_name = meta.get('video') or meta.get('video_name') or segment_id
        start = float(meta.get('start', 0.0))
        end = float(meta.get('end', 0.0))
        return video_name, start, end
    except Exception:
        return segment_id, 0.0, 0.0


def extract_keyword_queries_from_query(query: str) -> List[str]:
    # naive keywords extraction: keep words length>=3, deduplicate
    stop = {
        "the","and","with","then","that","have","this","from","into","what","which","when","were","will","like","left","right","top","bottom","front","rear","back","show","showing","shows","look","looks","looking","please","tell","explain","how","why","where","who","whom","whose","does","did","doing","done","make","made","press","pressed","pressing","step","first","second","third","final","begin","end","start","stop","turn","on","off"
    }
    tokens = [w.strip(" ,.;:!?()[]{}\"'\n\t").lower() for w in query.split()]
    candidates = []
    seen = set()
    for w in tokens:
        if len(w) >= 3 and w not in stop and w.isalpha() and w not in seen:
            candidates.append(w)
            seen.add(w)
    # fallback
    if not len(candidates):
        candidates = [query[:64]]
    return candidates[:8]


# ==================================================================================
# == YOLO-World Implementation
# ==================================================================================
try:
    from ultralytics import YOLO
    _HAS_YOLO = True
except Exception:
    _HAS_YOLO = False


def _safe_set_yolo_classes(yolo_model, keywords: list):
    """Safely call yolo_model.set_classes without causing device-mismatch errors.
    Strategy:
    - If the model is on CUDA, briefly move it to CPU, call set_classes, then move it back to the preferred device (GPU if available).
    - If move operations fail, fall back to calling set_classes directly and surface the exception.
    This keeps changes local and avoids editing upstream ultralytics code.
    """
    if yolo_model is None:
        return
    try:
        # try to detect current device from model parameters
        orig_device = None
        try:
            params = next(yolo_model.model.parameters())
            orig_device = params.device
        except Exception:
            try:
                params = next(yolo_model.parameters())
                orig_device = params.device
            except Exception:
                orig_device = None

        prefer_cuda = torch.cuda.is_available()
        # If model on CUDA, move to CPU for set_classes to avoid index_select device mismatch
        moved_to_cpu = False
        try:
            if orig_device is not None and orig_device.type == 'cuda':
                try:
                    yolo_model.to('cpu')
                    moved_to_cpu = True
                except Exception:
                    moved_to_cpu = False
        except Exception:
            moved_to_cpu = False

        # call set_classes (may create CPU tensors that would conflict with CUDA tensors)
        yolo_model.set_classes(keywords)

        # attempt to move model back to CUDA if available
        if prefer_cuda:
            try:
                yolo_model.to('cuda:0')
            except Exception:
                # best-effort, ignore failures
                pass
    except Exception:
        # re-raise to let caller handle/log as before
        raise

def detect_objects_for_segments_yolo_world(
    segment_ids: List[str],
    video_path_db: Any,
    video_segments_db: Any,
    pre_extracted_frames_db: Optional[Dict[str, Any]],
    num_frames: int,
    keywords: List[str], # 修改：接收关键词
    yolo_model: Any,
    max_frames_per_segment: int = 8
) -> Dict[str, str]:
    if not _HAS_YOLO:
        print("[Refine-DET] Skipped: ultralytics library is not installed.")
        return {}
    
    if yolo_model is None:
        print("[Refine-DET] YOLO-World model not pre-loaded. Skipping.")
        return {}

    if not keywords:
        print("[Refine-DET] No text queries provided for YOLO-World. Skipping.")
        return {}

    if not segment_ids:
        return {}
        
    # Set model classes based on dynamically generated keywords
    try:
        print(f"[YOLO-World] Setting model keywords: {keywords}")
        try:
            _safe_set_yolo_classes(yolo_model, keywords)
        except Exception as e:
            print(f"[YOLO-World][WARN] set_classes failed: {e}")
    except Exception:
        # keep outer flow stable
        pass

    seg2desc: Dict[str, str] = {}
    total_frames = 0
    total_detections = 0

    try:
        for s_id in tqdm(segment_ids, desc="Refine Pass: DET (YOLO-World)"):
            try:
                frames: List[Image.Image] = []
                # Frame extraction logic remains the same as OWL-ViT version
                # ... (Assuming this part is correct and handles pre_extracted_frames)
                if pre_extracted_frames_db and s_id in pre_extracted_frames_db:
                    # 使用所有目标帧进行检测（不截断）
                    raw_list = pre_extracted_frames_db.get(s_id) or []
                    paths = [item[0] if isinstance(item, (list, tuple)) else item for item in raw_list]
                    frames = [Image.open(p).convert("RGB") for p in paths]
                else:
                    video_name, start, end = _get_segment_time(video_segments_db, s_id)
                    video_path = video_path_db._data[video_name]
                    frames = _sample_frames(video_path, start, end, max(1, min(num_frames, 16)))
                
                if not frames:
                    continue

                total_frames += len(frames)
                
                # Batch inference with dynamic confidence backoff
                detected_lines: List[str] = []
                try:
                    conf0 = float(os.environ.get("YOLOWORLD_CONF_THR", "0.25") or 0.25)
                except Exception:
                    conf0 = 0.25
                try:
                    iou_thr = float(os.environ.get("YOLOWORLD_IOU_THR", "0.50") or 0.50)
                except Exception:
                    iou_thr = 0.50
                names_map = getattr(yolo_model, 'names', {}) or {}
                backoff = []
                for c in [conf0, 0.25, 0.20, 0.15, 0.10, 0.05]:
                    if c not in backoff and c >= 0.01:
                        backoff.append(c)
                for conf_thr in backoff:
                    # Ensure model on CUDA if available to speed up predictions and avoid CPU/CUDA tensor mixes
                    try:
                        if torch.cuda.is_available():
                            try:
                                yolo_model.to('cuda:0')
                            except Exception:
                                pass
                    except Exception:
                        pass

                    results = yolo_model.predict(source=frames, conf=conf_thr, iou=iou_thr, verbose=False)
                    detected_lines = []
                    for res in results:
                        boxes = getattr(res, 'boxes', None)
                        if boxes is None:
                            continue
                        for box in boxes:
                            try:
                                # ultralytics tensors -> python scalars
                                label_idx = int(getattr(box, 'cls', [0])[0])
                            except Exception:
                                try:
                                    label_idx = int(box.cls.item())
                                except Exception:
                                    label_idx = 0
                            label = names_map.get(label_idx, str(label_idx))
                            try:
                                score = float(getattr(box, 'conf', [0.0])[0])
                            except Exception:
                                try:
                                    score = float(box.conf.item())
                                except Exception:
                                    score = 0.0
                            if score < conf_thr:
                                continue
                            try:
                                x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
                            except Exception:
                                try:
                                    x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                                except Exception:
                                    x1 = y1 = x2 = y2 = 0
                            detected_lines.append(f"{label} ({score:.2f}) @ [{x1},{y1},{x2},{y2}]")
                    if detected_lines:
                        break
                
                if detected_lines:
                    uniq = sorted(list(set(detected_lines)))
                    seg2desc[s_id] = "; ".join(uniq[:20])
                    total_detections += len(uniq)
            
            except Exception as seg_err:
                print(f"[Refine-DET] Error processing segment {s_id} with YOLO-World: {seg_err}")
                continue

        print(f"[Refine-DET] segments={len(segment_ids)} frames={total_frames} detections={total_detections}")
    except Exception as e:
        print(f"[Refine-DET] A critical error occurred during the YOLO-World process: {e}")
        return seg2desc
        
    # If nothing detected at all, try a one-time fallback with generic, highly visible classes
    try:
        do_fallback = (total_detections == 0) and (os.environ.get("YOLOWORLD_FALLBACK_ON_EMPTY", "1").lower() in {"1","true","yes"})
        fallback_keywords = [
            "person", "car", "truck", "bus", "bicycle", "motorcycle",
            "flag", "banner", "microphone", "camera", "phone", "laptop",
            "screen", "table", "chair", "building", "gun", "police",
        ]
        if do_fallback and set([k.lower() for k in keywords]) != set([k.lower() for k in fallback_keywords]):
            print("[Refine-DET] No detections with smart keywords, retrying once with generic visible keywords...")
            return detect_objects_for_segments_yolo_world(
                segment_ids=segment_ids,
                video_path_db=video_path_db,
                video_segments_db=video_segments_db,
                pre_extracted_frames_db=pre_extracted_frames_db,
                num_frames=num_frames,
                keywords=fallback_keywords,
                yolo_model=yolo_model,
                max_frames_per_segment=max_frames_per_segment,
            )
    except Exception:
        pass

    return seg2desc


# ==================================================================================
# == (DEPRECATED) OWL-ViT Implementation
# ==================================================================================
# def detect_objects_for_segments_owlvit(...)
# ... (The entire old function is commented out or can be removed)
# ==================================================================================

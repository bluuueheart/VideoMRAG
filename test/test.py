import os
import sys
import json
import os
import sys
import json
import logging
import warnings
import multiprocessing
from datetime import datetime
import urllib.request
import subprocess
import glob
import argparse
from PIL import Image
import numpy as np
import base64
import traceback
import shutil
import re
import ast
import argparse
import pathlib

from faster_whisper import WhisperModel  # moved utility still needs class

# --- Ensure repo root (containing 'videorag/') is on sys.path ---
def _ensure_repo_root_on_syspath():
    try:
        cur = os.path.abspath(os.path.dirname(__file__))
        # Walk up a few levels to find a folder that directly contains 'videorag'
        for _ in range(6):
            if os.path.isdir(os.path.join(cur, "videorag")):
                if cur not in sys.path:
                    sys.path.insert(0, cur)
                break
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
    except Exception:
        # Best-effort; don't block execution
        pass

_ensure_repo_root_on_syspath()

from videorag.iterative_refinement import refine_context
from videorag.iterative_refiner import IterativeRefiner
from videorag._llm import get_default_external_llm_chat_model as get_default_ollama_chat_model

from video_urls import video_urls_multi_segment
from test_media_utils import (
    download_file,
    get_video_resolution,
    ensure_valid_video_or_skip,
    repair_mp4_faststart,
    is_valid_video,
    get_video_duration,
)
from test_env_utils import (
    sanitize_cuda_libs,
    check_dependencies,
    SimpleStore,
    check_models,
    normalize_question_text,
    extract_final_answer,
    choose_llm_config,
    image_to_base64,
    maybe_offline_fallback_answer,
)

# ---------------- Helper utilities moved to test_env_utils.py ----------------
# --- CRITICAL: Sanitize environment BEFORE importing torch-dependent libraries ---
sanitize_cuda_libs()
# Ensure user-local scripts (e.g. ~/.local/bin) are on PATH to avoid pip script warnings
try:
    from test_env_utils import ensure_user_local_bin_in_path
    ensure_user_local_bin_in_path()
except Exception:
    pass


try:
    import nest_asyncio
    _HAS_NEST_ASYNCIO = True
except Exception:
    _HAS_NEST_ASYNCIO = False

# ---- Notebook parity: keep the exact notebook lines ----
if _HAS_NEST_ASYNCIO:
    nest_asyncio.apply()

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)
# utility functions now imported from test_env_utils and test_media_utils
from processing_utils import _frame_cache_dir_for_segment, extract_frames_and_compress, transcribe_segment_audio
from question_processing import process_question

## duplicate choose_llm_config & helpers removed (now sourced from test_env_utils)




async def batch_main():
    # ---- Setup ----
    def _find_repo_root_from_here() -> str:
        cur = os.path.abspath(os.path.dirname(__file__))
        candidates = ("videorag", "faster-distil-whisper-large-v3")
        for _ in range(8):
            has_any = any(os.path.exists(os.path.join(cur, c)) for c in candidates)
            if has_any:
                return cur
            parent = os.path.dirname(cur)
            if parent == cur:
                break
            cur = parent
        return os.path.abspath(os.path.dirname(__file__))

    repo_root = _find_repo_root_from_here()
    work_dir = os.path.join(repo_root, "videorag-workdir", "batch_run")
    os.makedirs(work_dir, exist_ok=True)
    parser = argparse.ArgumentParser(description="Run VideoRAG batch processing.")
    parser.add_argument("file", nargs='?', default=None, help="Path to a specific JSON file with video segments to process.")
    parser.add_argument("--force", action="store_true", help="Force re-processing of all steps, ignoring caches.")
    parser.add_argument("--base-mode", action="store_true", help="Run in base mode, skipping iterative refinement.")
    parser.add_argument("--gpus", default=None, help="Comma-separated GPU ids to use for parallel workers, e.g. '0,1'. If omitted, runs single-process.")
    parser.add_argument("--workers", type=int, default=1, help="Maximum number of concurrent worker processes when --gpus is used.")
    # Note: VLM acceleration is enabled by default; no CLI flag required.
    args = parser.parse_args()
    # repo_root and work_dir already set above

    # ---- Preflight checks ----
    # sanitize_cuda_libs() is now called at the top of the script
    check_dependencies()
    check_models(repo_root)

    # ---- Initialize models ----
    print("[ASR] Loading faster-whisper model...")
    # Allow env override; else use repo_root/faster-distil-whisper-large-v3
    # Prefer explicit environment overrides; else default to user-provided shared model location or repo fallback
    # User-provided path (from your message)
    user_model_path = "/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/gaojinpeng02/00_opensource_models/huggingface.co/deepdml/faster-distil-whisper-large-v3.5"
    asr_model_path = (
        os.environ.get("FASTER_WHISPER_DIR")
        or os.environ.get("ASR_MODEL_PATH")
        or os.environ.get("DEFAULT_ASR_MODEL_PATH")
        or user_model_path
        or "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/00_opensource_models/huggingface.co/deepdml/faster-distil-whisper-large-v3.5"
        or os.path.join(repo_root, "faster-distil-whisper-large-v3")
    )
    if not os.path.exists(asr_model_path):
        print(f"[ASR] Local model path not found: {asr_model_path}")
        allow_online = str(os.environ.get("ALLOW_HF_DOWNLOADS", "0")).lower() in {"1", "true", "yes"}
        if allow_online:
            print("[ASR] ALLOW_HF_DOWNLOADS=1 -> falling back to Hugging Face model id 'distil-large-v3' (will download if needed)")
            asr_model_path = "distil-large-v3"
        else:
            print("[ASR] Online downloads are disabled. To allow automatic downloads set environment: ALLOW_HF_DOWNLOADS=1")
            print("[ASR] Or set FASTER_WHISPER_DIR / ASR_MODEL_PATH to a local model folder. Aborting ASR initialization.")
            raise SystemExit(1)
    
    # Force ASR to use GPU when CUDA devices are available
    cuda_available = False
    try:
        import torch
        cuda_available = torch.cuda.is_available() and torch.cuda.device_count() > 0
    except Exception:
        cuda_available = False

    if cuda_available:
        try:
            asr_model = WhisperModel(asr_model_path, device="cuda", compute_type="int8")
            print("[ASR] Using CUDA with int8 precision")
        except Exception as e:
            print(f"[ASR] CUDA initialization failed ({e}), falling back to CPU...")
            asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="int8")
    else:
        # No CUDA -> CPU
        asr_model = WhisperModel(asr_model_path, device="cpu", compute_type="int8")
        print("[ASR] CUDA not available; Using CPU with int8 precision")
    
    llm_cfg = choose_llm_config()
    # Ensure VLM acceleration is enabled by default (signal to backends)
    try:
        setattr(llm_cfg, 'vlm_accel', True)
    except Exception:
        pass

    # ---- Batch Processing ----
    # 优先环境变量，其次默认路径；支持命令行传入文件或目录
    # 改为读取用户指定的 Data 目录（包含 query 与 top5_segments 的 JSON）
    input_base_dir = os.environ.get("INPUT_BASE_DIR") or "/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/lx/Data"
    single_json_file: str | None = None
    if args.file:
        user_path = os.path.abspath(args.file)
        if os.path.isfile(user_path):
            single_json_file = user_path
            input_base_dir = os.path.dirname(user_path)
        elif os.path.isdir(user_path):
            input_base_dir = user_path
    force_rerun = args.force or (os.environ.get("FORCE_RERUN", "").strip().lower() in {"1", "true", "yes"})

    def _infer_model_tag(model_name: str) -> str:
        name = (model_name or "").strip()
        lname = name.lower()
        try:
            from videorag._llm import MODEL_NAME_TO_LOCAL_PATH
        except Exception:
            MODEL_NAME_TO_LOCAL_PATH = {}

        for k in (MODEL_NAME_TO_LOCAL_PATH or {}).keys():
            if k and k in lname:
                return k

        # fallback heuristics
        if "internvl" in lname:
            return "internvl"
        if "qwen" in lname:
            return "qwen"
        if "llama" in lname:
            return "llama"
        if "gemma" in lname:
            return "gemma"
        if "minicpm" in lname:
            return "minicpm"
        base = lname.split(":", 1)[0]
        if "/" in base or "\\" in base:
            base = base.replace("\\", "/").rstrip("/").split("/")[-1]
        return (base or "misc").replace("/", "_")

    # Determine active chat model (shortname or path) and route outputs to a model-specific subfolder
    active_chat_model = os.environ.get("OLLAMA_CHAT_MODEL", "").strip() or get_default_ollama_chat_model()
    model_tag = _infer_model_tag(active_chat_model)
    
    # 根据 base-mode 调整输出目录
    if args.base_mode:
        model_tag = f"{model_tag}_base"
        print(f"[LLM] Base mode active. Output will be saved under directory tag: {model_tag}")

    # Allow override via OUTPUT_BASE_DIR; else use target user's Result path
    env_out = os.environ.get("OUTPUT_BASE_DIR")
    if env_out:
        # Ensure env_out ends with the model_tag subdirectory for per-model separation
        if os.path.basename(os.path.normpath(env_out)) != model_tag:
            output_base_dir = os.path.join(env_out, model_tag)
        else:
            output_base_dir = env_out
    else:
        output_base_dir = f"/home/hadoop-aipnlp/dolphinfs_hdd_hadoop-aipnlp/KAI/gaojinpeng02/lx/Result/{model_tag}"
    current_mode = "base" if args.base_mode else "refine"
    
    # 如果目录不存在，不要在这里抛错；允许走顶层 JSON 模式/单文件模式
    video_id_dirs: list[str] = []
    if os.path.isdir(input_base_dir):
        video_id_dirs = [d for d in os.listdir(input_base_dir) if os.path.isdir(os.path.join(input_base_dir, d))]
    else:
        print(f"[Input] Base dir not found or not a directory: {input_base_dir}")

    # Fallback: 顶层JSON模式（当目录下没有子目录时，直接读取 *.json）
    if not video_id_dirs:
        from video_urls import video_urls_multi_segment as _video_map

        def _build_segment_urls_from_top5(top5_names: list[str]) -> dict[str, str]:
            seg_urls: dict[str, str] = {}
            for name in top5_names or []:
                try:
                    if not isinstance(name, str):
                        continue
                    parts = name.split('_')
                    # 形如 U2gvha4CipY_7 （全局第7个30秒片段）
                    if len(parts) == 2 and parts[1].isdigit():
                        base = parts[0]
                        seg_idx = int(parts[1])
                        three_min_idx = seg_idx // 6
                        thirty_idx = seg_idx % 6
                        video_info = _video_map.get(base)
                        if not video_info:
                            print(f"[Top5] Skip: video '{base}' not found in video_urls.py")
                            continue
                        clip_key = f"{base}_{three_min_idx}.mp4"
                        url = video_info.get(clip_key) or next(iter(video_info.values()), None)
                        if not url:
                            continue
                        # 若找不到三分钟分片键而回退到唯一文件(如 base.mp4)，标记 FULL 以便下游做全局偏移
                        if clip_key in video_info:
                            seg_id = f"{base}_{three_min_idx}_{thirty_idx}"
                        else:
                            seg_id = f"{base}FULL_{three_min_idx}_{thirty_idx}"
                        seg_urls[seg_id] = url
                    # 已是三元形式: base_三分钟序号_30秒序号
                    elif len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                        base = '_'.join(parts[:-2])
                        three_min_idx = int(parts[-2])
                        thirty_idx = int(parts[-1])
                        video_info = _video_map.get(base)
                        if not video_info:
                            print(f"[Top5] Skip: video '{base}' not found in video_urls.py")
                            continue
                        clip_key = f"{base}_{three_min_idx}.mp4"
                        url = video_info.get(clip_key) or next(iter(video_info.values()), None)
                        if not url:
                            continue
                        if clip_key in video_info:
                            seg_id = f"{base}_{three_min_idx}_{thirty_idx}"
                        else:
                            seg_id = f"{base}FULL_{three_min_idx}_{thirty_idx}"
                        seg_urls[seg_id] = url
                    else:
                        # 最宽松兼容：无法解析时，尝试把前缀当作 video id
                        base = parts[0]
                        video_info = _video_map.get(base)
                        if not video_info:
                            continue
                        url = next(iter(video_info.values()), None)
                        if not url:
                            continue
                        seg_urls[name] = url
                except Exception:
                    continue
            return seg_urls

        processed_json_files: set[str] = set()
        json_files = [single_json_file] if single_json_file else glob.glob(os.path.join(input_base_dir, "*.json"))
        if not json_files:
            print(f"[Top5] No JSON files found under: {input_base_dir}")
            return
        for json_file_path in json_files:
            norm_path = os.path.abspath(json_file_path)
            if norm_path in processed_json_files:
                continue
            processed_json_files.add(norm_path)

            print(f"\nProcessing file: {norm_path}")
            try:
                with open(norm_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON {norm_path}: {e}")
                continue

            # 仅处理包含 query + top5_segments 的结构（支持 query_id）
            if not (isinstance(qa_data, dict) and 'query' in qa_data and 'top5_segments' in qa_data):
                print(f"[Skip] Not a 'query+top5_segments' JSON: {json_file_path}")
                continue

            # 规范化问题文本
            question = normalize_question_text(qa_data.get('query', ''))

            # 根据 top5_segments 构建 segment -> url 映射
            segment_urls = _build_segment_urls_from_top5(qa_data.get('top5_segments') or [])
            if not segment_urls:
                print(f"[Skip] No resolvable segments in {json_file_path}")
                continue

            # 输出位置：使用 query_id.json 保存结果，若缺失则回退到原文件名
            qid = qa_data.get('query_id')
            if qid is None:
                relative_path = os.path.basename(json_file_path) if single_json_file else os.path.relpath(json_file_path, input_base_dir)
                output_file_path = os.path.join(output_base_dir, relative_path)
            else:
                rel_dir = os.path.dirname(os.path.relpath(json_file_path, input_base_dir)) if not single_json_file else ""
                if rel_dir in (".", ""):
                    output_file_path = os.path.join(input_base_dir, f"{qid}.json")
                else:
                    output_file_path = os.path.join(input_base_dir, rel_dir, f"{qid}.json")
            failure_log_path = output_file_path + ".failures.jsonl"

            # 读取已存在结果以便 resume
            existing_results = []
            existing_questions = set()
            if not force_rerun:
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                            if isinstance(existing_results, dict):
                                existing_results = [existing_results]
                    for item in existing_results:
                        if isinstance(item, dict) and "question" in item:
                            if item.get("mode") == current_mode:
                                existing_questions.add(item["question"])
                except Exception:
                    try:
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        os.rename(output_file_path, output_file_path + ".bak")
                        print(f"[Resume] Corrupted result file backed up: {output_file_path}.bak")
                    except Exception:
                        pass
                    existing_results = []

            def append_failure_top(idx: int, question_text: str, err: Exception):
                rec = {
                    "video_id": "_top5_mode_",
                    "result_rel_path": os.path.relpath(json_file_path, input_base_dir) if input_base_dir else os.path.basename(json_file_path),
                    "index": idx,
                    "question": question_text,
                    "error": str(err),
                    "traceback": traceback.format_exc(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as flog:
                        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    master_fail = os.path.join(output_base_dir, "_failures.jsonl")
                    with open(master_fail, "a", encoding="utf-8") as mlog:
                        mlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            if (not force_rerun) and (question in existing_questions):
                print(f"[Skip] {question[:50]}...")
                continue

            try:
                touched_paths_for_top: set = set()
                answer_obj = await process_question(
                    question,
                    segment_urls,
                    work_dir,
                    asr_model,
                    llm_cfg,
                    touched_paths=touched_paths_for_top,
                    base_mode=args.base_mode
                )
                clean_ans = extract_final_answer(answer_obj.get("answer", ""))
                output_data = ([] if force_rerun else existing_results[:]) + [{"question": question, "answer": clean_ans, "mode": current_mode}]
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
            except Exception as err:
                print(f"[Top5 Error] {err}")
                append_failure_top(0, str(question), err)

        return

    # 如果存在 video 子目录，按目录处理；支持使用 --gpus 启动多个子进程，每个子进程绑定单个 GPU
    for video_id in video_id_dirs:
        video_dir_path = os.path.join(input_base_dir, video_id)
        json_files = glob.glob(os.path.join(video_dir_path, "*.json"))
        touched_paths_for_video: set = set()
        
        # Determine segment URLs for the current video_id
        if video_id not in video_urls_multi_segment:
            print(f"SKIPPING: Video ID {video_id} not found in video_urls.py")
            continue
        
        video_info = video_urls_multi_segment[video_id]
        is_pre_segmented = any('_' in k for k in video_info.keys())
        
        segment_urls = {}
        if is_pre_segmented:
            # 有序号的视频：3分钟片段，拆分为30秒
            for filename, url in video_info.items():
                if '_' in filename and filename.split('_')[-1].split('.')[0].isdigit():
                    for i in range(6):
                        segment_name = f"{filename.split('.')[0]}_{i}"
                        segment_urls[segment_name] = url
                else:
                    segment_urls[filename] = url
            segment_urls = dict(list(segment_urls.items())[:5])
        else:
            # Not pre-segmented, create virtual segments
            main_video_url = list(video_info.values())[0]
            for i in range(5):
                segment_urls[f"{video_id}_{i}"] = main_video_url

        if not segment_urls:
            print(f"SKIPPING: No segments found for video ID {video_id}")
            continue

        processed_json_files: set[str] = set()
        for json_file_path in json_files:
            norm_path = os.path.abspath(json_file_path)
            if norm_path in processed_json_files:
                continue
            processed_json_files.add(norm_path)

            print(f"\nProcessing file: {norm_path}")
            try:
                with open(norm_path, 'r', encoding='utf-8') as f:
                    qa_data = json.load(f)
            except Exception as e:
                print(f"Error reading JSON {norm_path}: {e}")
                continue
            
            # Resolve output path first for resume & logging
            relative_path = os.path.relpath(json_file_path, input_base_dir)
            output_file_path = os.path.join(output_base_dir, relative_path)
            failure_log_path = output_file_path + ".failures.jsonl"

            # Resume: load existing results if any (unless force rerun)
            existing_results = []
            existing_questions = set()
            if not force_rerun:
                try:
                    if os.path.exists(output_file_path):
                        with open(output_file_path, "r", encoding="utf-8") as f:
                            existing_results = json.load(f)
                            if isinstance(existing_results, dict):
                                existing_results = [existing_results]
                    # build question -> pos map
                    for pos, item in enumerate(existing_results):
                        if isinstance(item, dict) and "question" in item:
                            if item.get("mode") == current_mode:
                                existing_questions.add(item["question"])
                except Exception:
                    # If corrupted, back it up and start fresh
                    try:
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                        os.rename(output_file_path, output_file_path + ".bak")
                        print(f"[Resume] Corrupted result file backed up: {output_file_path}.bak")
                    except Exception:
                        pass
                    existing_results = []

            def append_failure(idx: int, question_text: str, err: Exception):
                rec = {
                    "video_id": video_id,
                    "result_rel_path": relative_path,
                    "index": idx,
                    "question": question_text,
                    "error": str(err),
                    "traceback": traceback.format_exc(),
                    "ts": datetime.now().isoformat(timespec="seconds"),
                }
                try:
                    os.makedirs(os.path.dirname(failure_log_path), exist_ok=True)
                    with open(failure_log_path, "a", encoding="utf-8") as flog:
                        flog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    # Also write to a model-level master failures file
                    master_fail = os.path.join(output_base_dir, "_failures.jsonl")
                    with open(master_fail, "a", encoding="utf-8") as mlog:
                        mlog.write(json.dumps(rec, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            output_data = [] if force_rerun else existing_results[:]  # start from existing or reset when forced

            questions = qa_data if isinstance(qa_data, list) else [qa_data]
            
            for i, item in enumerate(questions):
                question = item.get("question")
                if not question:
                    continue
                
                # 规范化问题文本
                question = normalize_question_text(question)

                print(f"\n--- Processing Q{i}: {question} for video {video_id} ---")

                # Resume: skip if already completed (has question in existing results)
                if (not force_rerun) and (question in existing_questions):
                    print(f"[Skip] Q{i}: {question[:50]}...")
                    continue

                try:
                    # The process_question function encapsulates the logic for a single query
                    answer_obj = await process_question(
                        question,
                        segment_urls,
                        work_dir,
                        asr_model,
                        llm_cfg,
                        touched_paths=touched_paths_for_video,
                        base_mode=args.base_mode
                    )
                    record = {
                        "question": question,
                        "answer": extract_final_answer(answer_obj.get("answer", "")),
                        "mode": current_mode,
                    }
                    output_data.append(record)
                    # Persist progress incrementally after each success
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    with open(output_file_path, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                except Exception as err:
                    print(f"[Batch Error-Q{i}] {err}")
                    append_failure(i, str(question), err)
                    # Do not write a result record, so that next run retries this question

            # Final save (already incrementally saved after each success)
            try:
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                with open(output_file_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, ensure_ascii=False, indent=2)
                print(f"[Saved] {output_file_path}")
            except Exception as err:
                print(f"[Save Error] {err}")

        # After all JSONs for this video_id, check completeness; if all done and no pending gaps, clean caches
        try:
            all_done = True
            for jf in json_files:
                # Load input questions
                try:
                    with open(jf, "r", encoding="utf-8") as f:
                        qa_data = json.load(f)
                except Exception:
                    all_done = False
                    break
                questions = qa_data if isinstance(qa_data, list) else [qa_data]
                expected_n = len(questions)
                # Corresponding results
                relp = os.path.relpath(jf, input_base_dir)
                outp = os.path.join(output_base_dir, relp)
                if not os.path.exists(outp):
                    all_done = False
                    break
                try:
                    with open(outp, "r", encoding="utf-8") as f:
                        res_list = json.load(f)
                        if isinstance(res_list, dict):
                            res_list = [res_list]
                except Exception:
                    all_done = False
                    break
                present_questions = set()
                for item in res_list:
                    if isinstance(item, dict) and "question" in item:
                        present_questions.add(item["question"])
                # Require all questions present
                if len(present_questions) < expected_n:
                    all_done = False
                    break

            if all_done and touched_paths_for_video:
                removed = 0
                for p in list(touched_paths_for_video):
                    try:
                        if os.path.isdir(p):
                            shutil.rmtree(p, ignore_errors=True)
                            removed += 1
                        elif os.path.isfile(p):
                            os.remove(p)
                            removed += 1
                    except Exception:
                        pass
                print(f"[Cache] Cleaned {video_id} ({removed} items)")
        except Exception as err:
            print(f"[Cache] Skip cleaning for {video_id}: {err}")

    # --- Lightweight parallel launcher: spawn subprocess per video and bind CUDA_VISIBLE_DEVICES per child ---
    gpus_arg = args.gpus or os.environ.get("WORKER_GPUS")
    if gpus_arg:
        gpu_list = [g.strip() for g in str(gpus_arg).split(',') if g.strip()]
    else:
        gpu_list = []

    # Allow mapping of GPU -> model override via environment var MODEL_GPU_OVERRIDES like "0=ollama-model,1=internvl3_5-8b-hf"
    model_overrides = {}
    mo = os.environ.get("MODEL_GPU_OVERRIDES", "").strip()
    if mo:
        for part in mo.split(','):
            if '=' in part:
                g, m = part.split('=', 1)
                model_overrides[g.strip()] = m.strip()

    if gpu_list and video_id_dirs:
        procs = []
        for idx, video_id in enumerate(video_id_dirs):
            assigned_gpu = gpu_list[idx % len(gpu_list)]
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = assigned_gpu
            # If user provided model override for this GPU, set OLLAMA_CHAT_MODEL for the child
            if assigned_gpu in model_overrides:
                env['OLLAMA_CHAT_MODEL'] = model_overrides[assigned_gpu]
            # Always enable VLM_ACCEL for child workers by default
            env['VLM_ACCEL'] = '1'
            # Launch child process to handle single video directory; pass the directory as --file
            cmd = [sys.executable, os.path.abspath(__file__), os.path.join(input_base_dir, video_id)]
            if args.force:
                cmd.append('--force')
            if args.base_mode:
                cmd.append('--base-mode')
            p = subprocess.Popen(cmd, env=env)
            procs.append(p)

        # Wait for children
        for p in procs:
            p.wait()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    try:
        import asyncio
        asyncio.run(batch_main())
    except Exception as e:
        print(f"[Batch Error] {e}")



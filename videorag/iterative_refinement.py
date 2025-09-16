import json
import asyncio
import os
import re
from typing import List, Dict, Any, Optional
import math, statistics
from PIL import Image
import numpy as np
from types import SimpleNamespace
from ._config import (
    DEDUP_PHASH_THRESHOLD_DEFAULT,
    DEDUP_DEBUG,
    FRAME_COUNT_MAPPING_EXTENDED,
)
from .refine_frames_utils import (
    _map_score_to_frames,
    _ir_average_hash,
    _ir_hamming_int,
    _ir_frame_hash,
    _ir_hash_distance,
    _dedup_frames,
    _ensure_dir,
    _uniform_subsample,
    _extract_frame_opencv,
    _interpolate_and_fill_frames,
)
# Delay importing IterativeRefiner to runtime to avoid heavy/recursive imports at module import time

# ---------------------------------------------------------------
# 公共: 视觉 caption token 规范化 (词形简化 + 同义合并)
# 用途: 1) _simple_tokenize 2) diff summary 中生成去重短语
# 说明: 维持原先两处逻辑的并集, 保证行为一致, 方便后续单点维护。
# ---------------------------------------------------------------
def _normalize_visual_token(word: str) -> str:
    w = word.lower()
    # 同义/归并映射 (综合原 synonym_map + syn_core)
    synonym_map = {
        "people": "person", "persons": "person", "men": "man", "women": "woman", "kids": "child", "children": "child",
        "screens": "screen", "monitor": "screen", "monitors": "screen", "display": "screen", "displays": "screen", "screenshots": "screen",
        "cars": "car", "vehicle": "car", "vehicles": "car", "automobile": "car", "autos": "car",
        "bikes": "bike", "bicycle": "bike", "bicycles": "bike",
        "buttons": "button", "key": "button", "keys": "button",
        "computers": "computer", "laptops": "laptop",
        "phones": "phone", "smartphone": "phone", "smartphones": "phone",
        "menubar": "menu", "menus": "menu",
        "flags": "flag", "buildings": "building", "tables": "table",
        "persons": "person", "microphones": "microphone", "cameras": "camera",
    }
    if w in synonym_map:
        return synonym_map[w]
    # 规则后缀简化 (顺序与原实现保持)
    for suf, rep in [
        ("ingly", "ing"), ("sses", "ss"), ("ies", "y"), ("ing", ""), ("ized", "ize"), ("ises", "ise"),
        ("ied", "y"), ("ers", "er"), ("ives", "ife"), ("ves", "f"), ("s", "")
    ]:
        if len(w) > 4 and w.endswith(suf):
            stem = w[: -len(suf)] + rep
            if len(stem) >= 3:
                w = stem
                break
    # 过去式/过去分词 (与原逻辑一致)
    for suf in ["ed", "d"]:
        if len(w) > 4 and w.endswith(suf):
            base = w[: -len(suf)]
            if len(base) >= 3:
                w = base
                break
    return synonym_map.get(w, w)

# 统一的视觉类停用词（模块级常量，供全局复用）
STOPWORDS_VISUAL = {
    "the","and","with","then","that","have","this","from","into","what","which","when","were","will","like",
    "left","right","top","bottom","front","rear","back","show","showing","shows","look","looks","looking",
    "please","tell","explain","how","why","where","who","whom","whose","does","did","doing","done","make","made",
    "press","pressed","pressing","step","first","second","third","final","begin","end","start","stop","turn","on","off",
    "a","an","of","in","at","to","for","is","it","are","be","as","by","or"
}

def _simple_tokenize(text: str) -> list[tuple[str, str]]:
    """模块级、统一的可视 token 简易分词+规范化。
    返回 (原词, 规范词) 列表；依赖 _normalize_visual_token 并使用共享停用词。
    """
    raw_tokens = re.split(r"[^A-Za-z0-9]+", text.lower())
    out: list[tuple[str, str]] = []
    for tok in raw_tokens:
        if len(tok) < 3:
            continue
        if tok in STOPWORDS_VISUAL:
            continue
        norm = _normalize_visual_token(tok)
        if norm in STOPWORDS_VISUAL or len(norm) < 3:
            continue
        out.append((tok, norm))
    return out

async def refine_context(
    query: str,
    initial_context: List[Dict[str, Any]],
    config: Dict[str, Any],
    resources: Dict[str, Any],
    plan: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    print("\n--- [Refine] Starting context refinement process ---")
    
    if plan is None:
        print("[Refine] Step 1: No pre-computed plan found. Generating a new one...")
        # Local import to avoid circular/heavy imports at module import time
        from .iterative_refiner import IterativeRefiner
        refiner = IterativeRefiner(config, llm_api_func=None)
        plan = await refiner.plan(query, initial_context)
        print("[Refine] Step 1: Plan generated successfully.")
    else:
        # 复用外部传入 plan: 仅跳过规划, 后续仍执行补帧 / OCR / DET / diff summary
        print("[Refine] Step 1: Reusing provided plan (skip planning stage).")
    
    if plan.get("status") == "refine":
        print(f"[Refine] Step 2: Plan status is 'refine'. Proceeding with data extraction for {len(plan.get('targets', []))} targets.")
        targets = plan.get("targets", [])
        # record segments that failed or were missing during refinement so caller can retry later
        failures: list[dict] = []
        if not targets:
            print("[Refine] No specific targets to refine. Exiting.")
            return plan

        # 计算是否需要 OCR / DET
        ocr_params = any(t.get('refinement_params', {}).get('run_ocr') for t in targets)
        det_params = any(t.get('refinement_params', {}).get('run_det') for t in targets)
        # 外部可通过 config.force_refine_ocr / force_refine_det 强制开启（数值/图表类问题启发式触发）
        if not ocr_params and config.get('force_refine_ocr'):
            for t in targets:
                rp = t.setdefault('refinement_params', {})
                rp['run_ocr'] = True
            ocr_params = True
            print('[Refine] Modality override: force_refine_ocr=1 -> OCR enabled for all targets.')
        if not det_params and config.get('force_refine_det'):
            for t in targets:
                rp = t.setdefault('refinement_params', {})
                rp['run_det'] = True
            det_params = True
            print('[Refine] Modality override: force_refine_det=1 -> DET enabled for all targets.')
        print(f"[Refine] Modality check: OCR needed: {ocr_params}, DET needed: {det_params}")
    # ...existing code...
        # --- Frame de-dup + 插值补齐 (参数化阈值 + 扩展映射) ---
        pre_frames = resources.get("pre_extracted_frames")
        video_segments = resources.get("video_segments")
        video_path_db = resources.get("video_path_db")

        # 解包 SimpleStore -> dict，保证后续使用统一字典接口
        try:
            if hasattr(video_segments, "_data") and isinstance(video_segments._data, dict):
                video_segments = video_segments._data
        except Exception:
            pass
        try:
            if hasattr(video_path_db, "_data") and isinstance(video_path_db._data, dict):
                video_path_db = video_path_db._data
        except Exception:
            pass

        # 从 config 读取阈值 / 映射 / debug
        cfg_threshold = config.get("refine_phash_threshold", DEDUP_PHASH_THRESHOLD_DEFAULT)
        cfg_debug = bool(config.get("refine_dedup_debug", DEDUP_DEBUG))
        frame_mapping = config.get("frame_count_mapping") or FRAME_COUNT_MAPPING_EXTENDED

        if isinstance(pre_frames, dict):
            for tgt in targets:
                clip_id = tgt.get("clip_id")
                if not clip_id or clip_id not in pre_frames:
                    # missing pre-extracted frames for this clip -> mark failure and skip
                    if clip_id:
                        failures.append({"clip_id": clip_id, "reason": "no_pre_extracted_frames"})
                    continue

                # 评分字段（示例）：优先 refinement_params 中的 combined_score；否则 answerability+density；均需容错 None
                rparams = tgt.get("refinement_params", {}) or {}
                score_for_mapping = None
                comb = rparams.get("combined_score", None)
                if isinstance(comb, (int, float)):
                    score_for_mapping = float(comb)
                else:
                    ans = rparams.get("answerability_score", 0)
                    den = rparams.get("density_score", 0)
                    try:
                        score_for_mapping = float(ans or 0) + float(den or 0)
                    except Exception:
                        score_for_mapping = 0.0
                target_cnt = rparams.get("new_sampling_rate_per_30s")

                # 若业务希望“总分→映射→目标帧数”，则用映射覆盖 target_cnt
                if score_for_mapping is not None:
                    mapped = _map_score_to_frames(score_for_mapping, frame_mapping)
                    # 若已有显式 target_cnt 且更小/更大，你可决定是否覆盖；这里优先显式 target_cnt
                    # 注意: 之前使用 if not target_cnt 会把合法的 0 误判为未设置，导致被映射值覆盖
                    # 需求: 仅当 target_cnt 为 None (未显式指定) 时才采用映射结果
                    if target_cnt is None:
                        target_cnt = mapped

                frames_list = pre_frames.get(clip_id) or []
                if not frames_list:
                    # no frames available for this clip after lookup -> mark failure
                    failures.append({"clip_id": clip_id, "reason": "frames_list_empty"})
                    continue
                # 之前: 若 target_cnt 为 0/None 会整体跳过补帧, 造成去重后帧数不足无法插值补齐。
                # 现改: 当 target_cnt 为 None 时, 退化为去重后帧数, 仍走统一插值流程(可统一时间戳/结构)。
                if target_cnt is None:
                    target_cnt = len(frames_list)
                # 若明确指定 0 则保持跳过(极端策略), 仅当 target_cnt>0 才执行补帧。
                if target_cnt <= 0:
                    continue

                orig_cnt = len(frames_list)
                all_paths = [p for p, _ts in frames_list]

                dedup_only = _dedup_frames(
                    all_paths,
                    target_count=None,           # 仅去重，此处不补
                    threshold=cfg_threshold,
                    debug=cfg_debug
                )
                if len(dedup_only) != orig_cnt:
                    print(f"[Refine] Frame de-dup({clip_id}): {orig_cnt} -> {len(dedup_only)} (thr={cfg_threshold})")

                seg_meta = None
                if isinstance(video_segments, dict):
                    seg_meta = video_segments.get(clip_id)
                    # 统一元数据键名，保证存在 start_time/end_time 供插值使用
                    if seg_meta:
                        seg_meta = dict(seg_meta)
                        st = seg_meta.get("start_time", seg_meta.get("start"))
                        et = seg_meta.get("end_time", seg_meta.get("end"))
                        if isinstance(st, (int, float)):
                            seg_meta["start_time"] = float(st)
                        if isinstance(et, (int, float)):
                            seg_meta["end_time"] = float(et)
                    # 补充 video_path，供 OpenCV 插帧
                    if seg_meta and isinstance(video_path_db, dict):
                        vname = seg_meta.get("video") or seg_meta.get("video_name")
                        if vname and "video_path" not in seg_meta:
                            vp = video_path_db.get(vname)
                            if vp:
                                seg_meta["video_path"] = vp

                adjusted_pairs = _interpolate_and_fill_frames(
                    clip_id=clip_id,
                    dedup_paths=dedup_only,
                    orig_frames=frames_list,
                    target_cnt=target_cnt,
                    segment_meta=seg_meta,
                    video_path_db=video_path_db
                )

                if len(adjusted_pairs) != target_cnt:
                    print(f"[Refine][WARN] {clip_id} final={len(adjusted_pairs)} target={target_cnt} (可能缺少视频或 OpenCV)")
                    failures.append({"clip_id": clip_id, "reason": "insufficient_final_frames", "final": len(adjusted_pairs), "target": int(target_cnt)})

                pre_frames[clip_id] = adjusted_pairs

        ocr_map = {}
        det_map = {}
        updated_pre_frames_snapshot = None

        # 记录一次处理后的快照，供上层用于最终帧输入
        try:
            updated_pre_frames_snapshot = {k: list(v) for k, v in (pre_frames or {}).items()}
        except Exception:
            updated_pre_frames_snapshot = pre_frames

        if ocr_params or det_params:
            from ._videoutil.refinement_utils import extract_ocr_text_for_segments, detect_objects_for_segments_yolo_world
            video_path_db = resources.get("video_path_db")
            video_segments = resources.get("video_segments")
            pre_frames = resources.get("pre_extracted_frames")

            # 同样解包 SimpleStore，确保下游使用 dict.get()
            try:
                if hasattr(video_segments, "_data") and isinstance(video_segments._data, dict):
                    video_segments = video_segments._data
            except Exception:
                pass
            try:
                if hasattr(video_path_db, "_data") and isinstance(video_path_db._data, dict):
                    video_path_db = video_path_db._data
            except Exception:
                pass

            det_model = None
            if det_params:
                print("[Refine] Step 2a: DET required. Pre-loading YOLO-World model...")
                try:
                    from ultralytics import YOLO
                    # Prefer environment variable, fallback to mapping from _llm (if available)
                    model_path = os.environ.get("YOLOWORLD_MODEL_PATH", "")
                    if not model_path:
                        # try to import mapping from _llm, else fallback to central config
                        try:
                            from ._llm import MODEL_NAME_TO_LOCAL_PATH as _map
                            model_path = _map.get("yolov8m-worldv2", "") or _map.get("yolov8m-worldv2.pt", "")
                        except Exception:
                            model_path = ""
                        if not model_path:
                            try:
                                from ._config import YOLOWORLD_MODEL_PATH
                                model_path = YOLOWORLD_MODEL_PATH
                            except Exception:
                                pass

                    if not model_path or not os.path.exists(model_path):
                        print(f"[Refine-DET][ERROR] Model not found at {model_path}. Skipping DET.")
                        det_params = False
                    else:
                        det_model = YOLO(model_path)
                        print("[Refine] Step 2a: YOLO-World model pre-loaded successfully.")
                except Exception as e:
                    print(f"[Refine-DET][ERROR] Failed to pre-load YOLO-World model: {e}. Skipping DET.")
                    det_params = False

            if ocr_params:
                print("[Refine] Step 2b: OCR required. Running EasyOCR extraction...")
                try:
                    # 仅对标记 run_ocr 的目标做 OCR
                    segment_ids = [
                        t.get("clip_id")
                        for t in targets
                        if t.get("clip_id") and (t.get("refinement_params") or {}).get("run_ocr")
                    ]
                    requested_ocr_segment_ids = list(segment_ids)
                    ocr_frame_cnt = int(config.get("refine_ocr_frames", 6) or 6)
                    ocr_map = extract_ocr_text_for_segments(
                        segments=segment_ids,
                        video_path_db=SimpleNamespace(_data=video_path_db) if isinstance(video_path_db, dict) else video_path_db,
                        video_segments=video_segments,
                        num_frames=ocr_frame_cnt,
                        languages=None,
                        pre_extracted_frames=pre_frames,
                    ) or {}
                    print(f"[Refine] Step 2b: OCR extracted for {len(ocr_map)} segments.")
                    # mark failures for requested OCR segments that produced no result
                    for sid in (requested_ocr_segment_ids or []):
                        if sid and sid not in (ocr_map or {}):
                            failures.append({"clip_id": sid, "reason": "ocr_no_result"})
                except Exception as e:
                    print(f"[Refine-OCR][ERROR] OCR extraction failed: {e}")

            if det_params and det_model:
                print("[Refine] Step 2c: DET required. Generating smart visual keywords with LLM...")
                from .iterative_refiner import IterativeRefiner
                refiner = IterativeRefiner(config, llm_api_func=None)
                det_keywords = await refiner._generate_visual_keywords(query, initial_context)
                print(f"[Refine] Step 2c: Keywords generated: {det_keywords}. Running YOLO-World detection...")
                try:
                    # 仅对标记 run_det 的目标做 DET
                    segment_ids = [
                        t.get("clip_id")
                        for t in targets
                        if t.get("clip_id") and (t.get("refinement_params") or {}).get("run_det")
                    ]
                    requested_det_segment_ids = list(segment_ids)
                    det_frame_cnt = int(config.get("refine_det_frames", 8) or 8)
                    det_map = detect_objects_for_segments_yolo_world(
                        segment_ids=segment_ids,
                        video_path_db=SimpleNamespace(_data=video_path_db) if isinstance(video_path_db, dict) else video_path_db,
                        video_segments_db=video_segments,
                        pre_extracted_frames_db=pre_frames,
                        num_frames=det_frame_cnt,
                        keywords=det_keywords or [],
                        yolo_model=det_model,
                        max_frames_per_segment=det_frame_cnt,
                    ) or {}
                    print(f"[Refine] Step 2c: DET produced descriptions for {len(det_map)} segments.")
                    for sid in (requested_det_segment_ids or []):
                        if sid and sid not in (det_map or {}):
                            failures.append({"clip_id": sid, "reason": "det_no_result"})
                except Exception as e:
                    print(f"[Refine-DET][ERROR] Detection failed: {e}")

        print("[Refine] Step 3: Attaching refinement results to the plan.")
        plan["refinement_results"] = {
            "ocr_text_map": ocr_map,
            "det_text_map": det_map,
            "updated_pre_frames": (updated_pre_frames_snapshot or {})
        }

        # ------------------------------------------------------------------
        # 新增 Step 4: 视觉 -> 文本 多帧差异收敛摘要 (diff-based incremental summary)
        # 需求:
        #  1) 对 refined 后的多帧逐帧生成 caption
        #  2) 与上一帧 caption 做“差异描述”：去掉重复 token，仅保留新增对象/动作词
        #  3) 合并后对每个片段摘要裁剪到 N 字符 (默认 160，可通过 config['refine_diff_summary_max_chars'] 配置)
        # 说明:
        #  - 复用 MiniCPM 模型 (与 caption.py 中一致)；若模型不可用/出错则 graceful fallback
        #  - 为减少开销：每段最多处理 M 帧 (config['refine_diff_max_frames']，默认 <=12)
        #  - Token 粒度: 简单英文分词 + 去停用词 + 去重；不依赖额外 NLP 库，保证最小侵入
        # ------------------------------------------------------------------

        # 使用模块级统一实现的 _simple_tokenize

        async def _caption_single_image(model, tokenizer, image_path: str) -> str:
            """对单帧生成简洁 caption。失败返回空串。"""
            try:
                from PIL import Image
                img = Image.open(image_path).convert("RGB")
                prompt = "Describe the visual content of this frame in concise English (objects and actions only)."
                msgs = [{"role": "user", "content": prompt}]
                params = {
                    "use_image_id": False,
                    "max_slice_nums": 1,
                    "max_new_tokens": 96,
                    "temperature": 0.1,
                }
                import torch
                # Determine whether all model parameters live on a single CUDA device
                try:
                    from ._videoutil.multi_gpu import model_device_set
                    devs = model_device_set(model)
                except Exception:
                    try:
                        devs = {str(p.device) for p in model.parameters()}
                    except Exception:
                        devs = set()

                use_autocast = False
                if any(d.startswith('cuda') for d in devs):
                    cuda_devs = {d for d in devs if d.startswith('cuda')}
                    if len(cuda_devs) == 1:
                        use_autocast = True

                if use_autocast:
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
                        cap = model.chat(image=[img], msgs=msgs, tokenizer=tokenizer, **params)
                else:
                    with torch.inference_mode():
                        cap = model.chat(image=[img], msgs=msgs, tokenizer=tokenizer, **params)
                return (cap or "").replace("\n", " ").strip()
            except Exception:
                return ""

        def _safe_truncate_at_word_boundary(text: str, max_chars: int) -> str:
            """安全按词边界截断，避免切断半个词。"""
            if not text or max_chars <= 0:
                return ""
            if len(text) <= max_chars:
                return text
            cut = text[:max_chars]
            if cut and cut[-1].isalnum():
                i = len(cut) - 1
                while i >= 0 and cut[i].isalnum():
                    i -= 1
                cut = cut[: i + 1] if i >= 0 else ""
            return cut.rstrip('; ,')

        async def _build_diff_summary_for_segment(clip_id: str, frame_pairs: list[tuple[str,float]], max_chars: int, max_frames: int, model, tokenizer, decay_window: int) -> str:
            if not frame_pairs:
                return ""
            # 按时间排序
            frame_pairs_sorted = sorted(frame_pairs, key=lambda x: x[1])
            if max_frames > 0 and len(frame_pairs_sorted) > max_frames:
                n = len(frame_pairs_sorted)
                uniform_ratio = 0.6
                base_count = max(1, min(max_frames, int(round(max_frames * uniform_ratio))))
                base_idxs = np.linspace(0, n - 1, base_count, endpoint=True).astype(int).tolist()

                hashes = []
                for p, _ts in frame_pairs_sorted:
                    h, ok = _ir_frame_hash(p)
                    hashes.append(h if ok else None)
                change_scores: list[tuple[int, int]] = []
                for i in range(1, n):
                    d = _ir_hash_distance(hashes[i], hashes[i - 1])
                    change_scores.append((int(d), i))
                change_scores.sort(key=lambda x: x[0], reverse=True)

                chosen = set(base_idxs)
                for _score, idx in change_scores:
                    if len(chosen) >= max_frames:
                        break
                    chosen.add(idx)
                if len(chosen) < max_frames:
                    full_uniform = np.linspace(0, n - 1, max_frames, endpoint=True).astype(int).tolist()
                    for idx in full_uniform:
                        if len(chosen) >= max_frames:
                            break
                        chosen.add(idx)

                idxs = sorted(chosen)
                frame_pairs_sorted = [frame_pairs_sorted[i] for i in idxs]

            last_seen: dict[str, int] = {}
            pieces: list[str] = []
            for fi, (path, _ts) in enumerate(frame_pairs_sorted):
                cap = await _caption_single_image(model, tokenizer, path)
                if not cap:
                    continue
                toks = _simple_tokenize(cap)
                new_norms: list[str] = []
                for _orig, norm in toks:
                    prev = last_seen.get(norm)
                    if prev is None or fi - prev >= max(0, int(decay_window)):
                        new_norms.append(norm)
                for _orig, norm in toks:
                    last_seen[norm] = fi
                if not new_norms:
                    continue
                ordered_phrase_parts = []
                for word in cap.split():
                    lw = ''.join(ch for ch in word.lower() if ch.isalnum())
                    if lw:
                        wtmp = _normalize_visual_token(lw)
                        if wtmp in new_norms and lw not in ordered_phrase_parts:
                            ordered_phrase_parts.append(lw)
                if not ordered_phrase_parts:
                    continue
                phrase = ", ".join(ordered_phrase_parts)
                pieces.append(phrase)
                joined = "; ".join(pieces)
                if len(joined) >= max_chars:
                    return _safe_truncate_at_word_boundary(joined, max_chars)
            return _safe_truncate_at_word_boundary("; ".join(pieces), max_chars)

        diff_max_chars = int(config.get("refine_diff_summary_max_chars", 160) or 160)
        diff_max_frames = int(config.get("refine_diff_max_frames", 12) or 12)
        diff_decay_window = int(config.get("refine_diff_decay_window", 3) or 3)

        # Diff summary generation disabled in this deployment.
        # Do not attempt to load MiniCPM or build diff summaries to avoid
        # runtime errors when local model or torch variable is not available.
        diff_summaries: dict[str,str] = {}
        if isinstance(pre_frames, dict) and targets:
            # Intentionally skip building diff summaries; leave empty dict so
            # downstream code that expects the key continues to operate.

            plan["refinement_results"]["diff_summaries"] = diff_summaries
        if diff_summaries:
            print(f"[Refine] Step 2d: Generated diff summaries for {len(diff_summaries)} segments (<= {diff_max_chars} chars each).")
        else:
            print("[Refine] Step 2d: No diff summaries generated (model missing or no frames).")
        # expose failures (missing frames / insufficient frames / ocr/det misses) to caller
        try:
            plan.setdefault("refinement_results", {})["failures"] = failures
            if failures:
                print(f"[Refine] Failures recorded for {len(failures)} segments.")
        except Exception:
            pass
    else:
        print("[Refine] Plan status is 'final'. No data extraction needed.")
    
    # 追加: 打印三个组件 token 长度 (OCR / DET / DIFF) 以与 prompt builder 对齐
    try:
        rr = plan.get("refinement_results", {})
        ocr_tokens = sum(len((v or "").split()) for v in (rr.get("ocr_text_map") or {}).values())
        det_tokens = sum(len((v or "").split()) for v in (rr.get("det_text_map") or {}).values())
        diff_tokens = sum(len((v or "").split()) for v in (rr.get("diff_summaries") or {}).values())
        # 新增: 估算 Keyframes(base64) token 数
        kf_tokens = 0
        try:
            pre_frames_db = resources.get("pre_extracted_frames") or {}
            if isinstance(pre_frames_db, dict):
                # 缺省按 base64 长度 ~= 4/3 * bytes, 分词近似 1 token / 4 chars => tokens ≈ bytes / 3
                tokens_per_byte = float(config.get("keyframe_tokens_per_byte", 1.0 / 3.0) or (1.0 / 3.0))
                # 仅统计当前 plan 涉及的片段；若缺失则退化统计全部已缓存片段
                clip_ids = [t.get("clip_id") for t in plan.get("targets", []) if t.get("clip_id")]
                if not clip_ids:
                    clip_ids = list(pre_frames_db.keys())
                seen_paths = set()
                total_bytes = 0
                for cid in clip_ids:
                    frames = pre_frames_db.get(cid) or []
                    for it in frames:
                        try:
                            path = it[0] if isinstance(it, (list, tuple)) and it else None
                            if not path or path in seen_paths:
                                continue
                            seen_paths.add(path)
                            if os.path.exists(path):
                                total_bytes += os.path.getsize(path)
                        except Exception:
                            continue
                kf_tokens = int(total_bytes * tokens_per_byte)
        except Exception:
            kf_tokens = 0

        # 回写到 plan 以便后续使用
        plan.setdefault("refinement_results", {})["keyframe_tokens_estimate"] = kf_tokens

        print(f"[Refine] Token lengths -> OCR: {ocr_tokens}, DET: {det_tokens}, DIFF: {diff_tokens}, KEYFRAMES(est): {kf_tokens}")
    except Exception as _tok_err:
        print(f"[Refine][WARN] Token length stats failed: {_tok_err}")

    print("--- [Refine] Context refinement process finished ---")
    return plan

if __name__ == '__main__':
    mock_query = "Show me the full process of shutting down the primary machine, including pressing the final button."
    mock_initial_context = [
        {"id": "video1_2", "summary": "A person is approaching a large industrial machine.", "start_time": 30, "end_time": 35},
        {"id": "video1_4", "summary": "The person's hand reaches out towards a control panel.", "start_time": 60, "end_time": 65},
    ]
    mock_config = {}
    mock_resources = {
        "pre_extracted_frames": {
            # 示例: "video1_2": [("path/frame_0001.jpg", 30.1), ("path/frame_0002.jpg", 30.6), ...]
        },
        "video_path_db": None,
        "video_segments": {}
    }

    async def main():
        plan = await refine_context(mock_query, mock_initial_context, mock_config, mock_resources)
        print("\n--- Refinement Plan ---")
        print(json.dumps(plan, indent=2))

    asyncio.run(main())

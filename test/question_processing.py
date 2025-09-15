import os
from typing import List, Dict, Any, Set

from videorag.iterative_refinement import refine_context
from videorag.iterative_refiner import IterativeRefiner
from test_media_utils import download_file, ensure_valid_video_or_skip
from test_env_utils import SimpleStore
from segment_caption import generate_segment_caption
from test_env_utils import image_to_base64
from processing_utils import _frame_cache_dir_for_segment, extract_frames_and_compress, transcribe_segment_audio
from llm_prompt_builder import build_and_call_llm


def _parse_segment_timing(segment_id: str):
    """Parse start/end (30s window) from segment id with robust fallbacks."""
    try:
        if '_' in segment_id and segment_id.count('_') >= 2:
            parts = segment_id.split('_')
            if len(parts) >= 3 and parts[-1].isdigit() and parts[-2].isdigit():
                base = '_'.join(parts[:-2])
                three_idx = int(parts[-2])
                thirty_idx = int(parts[-1])
                if base.endswith('FULL'):
                    video_name = base[:-4]
                    start_time = three_idx * 180 + thirty_idx * 30
                else:
                    video_name = f"{base}_{three_idx}"
                    start_time = thirty_idx * 30
                end_time = start_time + 30
                return video_name, start_time, end_time
    except Exception:
        pass
    try:
        video_name, idx = segment_id.rsplit('_', 1)
        start_time = int(idx) * 30
        end_time = start_time + 30
        return video_name, start_time, end_time
    except Exception:
        return segment_id, 0, 30


def _neighbor_ids(clip_id: str) -> list[str]:
    try:
        base, idx = clip_id.rsplit('_', 1)
        i = int(idx)
        return [f"{base}_{i-1}", f"{base}_{i+1}"]
    except Exception:
        return []


async def process_question(query: str, segment_urls: Dict[str, str], work_dir: str, asr_model, llm_cfg,
                           touched_paths: Set[str] | None = None, base_mode: bool = False):
    # Step 1: 基础 5 段载入
    all_segment_data: List[Dict[str, Any]] = []
    video_segments_db: Dict[str, Dict[str, Any]] = {}
    video_path_db: Dict[str, str] = {}
    # 当 base_mode 为 True 时，后续提取的帧会统一缩放到 480x360 以压缩传输体积
    fs_force = (480, 360) if base_mode else None

    # Local benchmark directory override: if env var BENCHMARK_VIDEO_DIR is set
    # (or default path), prefer reading pre-downloaded full videos named <video_name>.mp4
    bench_dir = os.environ.get('BENCHMARK_VIDEO_DIR', '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/KAI/gaojinpeng02/lx/huggingface.co/datasets/Xingonearth/Benckmark_video/BenchmarkVideo_new')
    for segment_id, url in segment_urls.items():
        # Try to infer video file name from segment id (video_name from timing parser)
        video_name_guess, _, _ = _parse_segment_timing(segment_id)
        local_path = os.path.join(bench_dir, f"{video_name_guess}.mp4")
        if os.path.exists(local_path):
            vp = local_path
            print(f"[Local] Using local video for {segment_id}: {vp}")
        else:
            vp = download_file(url, work_dir)
        if not vp:
            continue
        vp = ensure_valid_video_or_skip(url, work_dir, vp)
        if not vp:
            continue
        if touched_paths is not None:
            touched_paths.add(vp)
        video_name, start_time, end_time = _parse_segment_timing(segment_id)
        # 根据 base_mode 传入 force_size，以便在 base 模式下得到统一压缩尺寸
        frames_with_ts, _ = extract_frames_and_compress(vp, work_dir, start_time, end_time, force_size=fs_force)
        if touched_paths is not None:
            try:
                frame_dir = _frame_cache_dir_for_segment(vp, work_dir, start_time, end_time)
                touched_paths.add(frame_dir)
            except Exception:
                pass
        transcript = transcribe_segment_audio(vp, asr_model, start_time, end_time)
        video_path_db[video_name] = vp
        video_segments_db[segment_id] = {"video": video_name, "start": float(start_time), "end": float(end_time), "video_path": vp}
        all_segment_data.append({
            'id': segment_id,
            'transcript': transcript,
            'timestamp': f"{start_time}s - {end_time}s",
            'frames_with_ts': frames_with_ts,
            'video_name': video_name,
            'start_time': start_time,
            'end_time': end_time,
        })
    # ...existing code...

    # 初始 caption (5 帧) - base_mode 下跳过 caption 生成
    for seg in all_segment_data:
        frames = [p for p, _ in seg.get('frames_with_ts')][:5]
        if frames:
            while len(frames) < 5:
                frames.append(frames[-1])
        if base_mode:
            # 不生成 caption，保留字段为空
            seg['segment_caption'] = ''
            seg['caption_frame_count'] = 0
            seg['caption_source'] = 'base'
        else:
            try:
                seg['segment_caption'] = await generate_segment_caption(frames, llm_cfg)
                seg['caption_frame_count'] = len(frames)
                seg['caption_source'] = 'base'
            except Exception:
                seg['segment_caption'] = ''
                seg['caption_frame_count'] = 0
                seg['caption_source'] = 'base'

    initial_context = [{
        'id': s['id'],
        'summary': s['transcript'],
        'start_time': s['start_time'],
        'end_time': s['end_time'],
        'frames': [p for p, _ in s.get('frames_with_ts')][:5],
        'caption': s.get('segment_caption', '')
    } for s in all_segment_data]

    resources = {"video_path_db": SimpleStore(video_path_db), "video_segments": SimpleStore(video_segments_db)}

    # Step 2: 规划 (带 temporal 字段)
    force_refine_ocr = False
    q_low = (query or '').lower()
    if any(ch.isdigit() for ch in q_low) or any(k in q_low for k in ['graph','chart','temperature','rate','percent','%','fluctuation','变化','曲线','图表','温度']):
        force_refine_ocr = True
    refinement_plan = None
    if not base_mode:
        cfg = {"fine_num_frames_per_segment": 15}
        if force_refine_ocr:
            cfg['force_refine_ocr'] = True
            cfg['fine_num_frames_per_segment'] = 20
        refiner = IterativeRefiner(config=cfg)
        refinement_plan = await refiner.plan(query, initial_context)
    else:
        print('[BaseMode] skip planning.')

    # Step 2.x: 邻居扩展
    extra_segments: List[Dict[str, Any]] = []
    neighbor_base_id: str | None = None  # 记录用于找邻居的中心段 id，便于最终排序
    if refinement_plan:
        total_score = (refinement_plan.get('scores') or {}).get('total', 0)
        temporal_flag = bool(refinement_plan.get('temporal_sequence_incomplete'))
        temporal_focus = refinement_plan.get('temporal_focus_clip_id') or ''
        status = refinement_plan.get('status')
        if status == 'final' and all_segment_data and (total_score > 5 or (temporal_flag and temporal_focus)):
            import re
            q_tokens = {t for t in re.split(r"[^a-z0-9\u4e00-\u9fa5]+", q_low) if len(t) >= 3}
            if temporal_flag and temporal_focus:
                base_seg = next((s for s in all_segment_data if s['id'] == temporal_focus), all_segment_data[0])
            else:
                def _rel(seg):
                    txt = (seg.get('transcript','') + ' ' + seg.get('segment_caption','')).lower()
                    return sum(1 for t in q_tokens if t and t in txt)
                base_seg = max(all_segment_data, key=_rel)
            neighbor_base_id = base_seg['id']
            for nid in _neighbor_ids(base_seg['id']):
                if not nid or nid in {s['id'] for s in all_segment_data}: continue
                url = segment_urls.get(nid)
                if not url: continue
                # neighbor segments: same local-first logic
                _vn_guess, _, _ = _parse_segment_timing(nid)
                local_path2 = os.path.join(bench_dir, f"{_vn_guess}.mp4")
                if os.path.exists(local_path2):
                    vp2 = local_path2
                    print(f"[Local] Using local neighbor video for {nid}: {vp2}")
                else:
                    vp2 = download_file(url, work_dir)
                if not vp2: continue
                vp2 = ensure_valid_video_or_skip(url, work_dir, vp2)
                if not vp2: continue
                # 使用与初始 5 段一致的 30s 解析逻辑而不是固定 0-30
                _vn2, _st2, _et2 = _parse_segment_timing(nid)
                frames2, _ = extract_frames_and_compress(vp2, work_dir, _st2, _et2, num_frames=5, force_size=(480,360) if base_mode else None)
                trans2 = transcribe_segment_audio(vp2, asr_model, _st2, _et2)
                cap_paths = [p for p,_ in frames2][:5]
                # 在 base_mode 下不生成 caption
                caption2 = '' if base_mode else (await generate_segment_caption(cap_paths, llm_cfg) if cap_paths else '')
                extra_segments.append({
                    'id': nid,'transcript': trans2,'timestamp':f'{_st2}s - {_et2}s','frames_with_ts': frames2,'video_name': _vn2,
                    'start_time':_st2,'end_time':_et2,'segment_caption': caption2,'caption_frame_count': (0 if base_mode else len(cap_paths)),'caption_source':'neighbor_final'
                })
        elif status == 'refine' and temporal_flag and total_score <= 5 and temporal_focus:
            target_entry = None
            for t in (refinement_plan.get('targets') or []):
                if t.get('clip_id') == temporal_focus:
                    target_entry = t; break
            neighbor_frames = None
            if target_entry:
                rp = target_entry.get('refinement_params') or {}
                neighbor_frames = rp.get('new_sampling_rate_per_30s')
            neighbor_base_id = temporal_focus
            for nid in _neighbor_ids(temporal_focus):
                if not nid or nid in {s['id'] for s in all_segment_data}: continue
                url = segment_urls.get(nid)
                if not url: continue
                _vn_guess, _, _ = _parse_segment_timing(nid)
                local_path2 = os.path.join(bench_dir, f"{_vn_guess}.mp4")
                if os.path.exists(local_path2):
                    vp2 = local_path2
                    print(f"[Local] Using local neighbor video for {nid}: {vp2}")
                else:
                    vp2 = download_file(url, work_dir)
                if not vp2: continue
                vp2 = ensure_valid_video_or_skip(url, work_dir, vp2)
                if not vp2: continue
                _vn2, _st2, _et2 = _parse_segment_timing(nid)
                frames2, _ = extract_frames_and_compress(vp2, work_dir, _st2, _et2, num_frames=neighbor_frames, force_size=(480,360) if base_mode else None)
                trans2 = transcribe_segment_audio(vp2, asr_model, _st2, _et2)
                cap_paths = [p for p,_ in frames2][:5]
                # 在 base_mode 下不生成 caption
                caption2 = '' if base_mode else (await generate_segment_caption(cap_paths, llm_cfg) if cap_paths else '')
                extra_segments.append({
                    'id': nid,'transcript': trans2,'timestamp':f'{_st2}s - {_et2}s','frames_with_ts': frames2,'video_name': _vn2,
                    'start_time':_st2,'end_time':_et2,'segment_caption': caption2,'caption_frame_count': (0 if base_mode else len(cap_paths)),'caption_source':'neighbor_temporal'
                })
            # Ensure neighbors are also scheduled for refinement with same params as the focused target
            if target_entry:
                rp = target_entry.get('refinement_params') or {}
                existing_targets = {t.get('clip_id') for t in (refinement_plan.get('targets') or [])}
                for seg in list(extra_segments):
                    nid = seg.get('id')
                    if nid and nid not in existing_targets:
                        refinement_plan.setdefault('targets', []).append({'clip_id': nid, 'refinement_params': rp})
        if extra_segments:
            print(f"[Neighbor] added {len(extra_segments)} segments -> total {len(all_segment_data)+len(extra_segments)}")
            all_segment_data.extend(extra_segments)
    else:
        print('[Info] no plan; skip neighbor.')

    # Step 2.5/2.6: 执行 refine（仅计划内 targets）
    refinement_plan_with_results = None
    if refinement_plan and refinement_plan.get('status') == 'refine' and not base_mode:
        target_ids = {t.get('clip_id') for t in (refinement_plan.get('targets') or []) if t.get('clip_id')}
        pre_extracted_frames = {seg['id']: seg.get('frames_with_ts') for seg in all_segment_data if seg['id'] in target_ids}
        def _is_under(root: str, p: str) -> bool:
            try:
                root_abs = os.path.abspath(root)
                p_abs = os.path.abspath(p)
                return os.path.commonprefix([p_abs + os.sep, root_abs + os.sep]) == root_abs + os.sep
            except Exception:
                return True
        sanitized = {}
        for cid, frs in pre_extracted_frames.items():
            clean = []
            for fp, ts in (frs or []):
                if fp and os.path.exists(fp) and _is_under(work_dir, fp):
                    clean.append((fp, float(ts)))
            if clean: sanitized[cid] = clean
        res2 = dict(resources); res2['pre_extracted_frames'] = sanitized
        refine_cfg = {"fine_num_frames_per_segment": 15}
        if force_refine_ocr:
            refine_cfg['force_refine_ocr'] = True
            refine_cfg['fine_num_frames_per_segment'] = 20
        refinement_plan_with_results = await refine_context(query=query, initial_context=initial_context, config=refine_cfg, resources=res2, plan=refinement_plan)
        ref_res = refinement_plan_with_results.get('refinement_results', {})
        ocr_map = ref_res.get('ocr_text_map', {})
        det_map = ref_res.get('det_text_map', {})
        diff_map = ref_res.get('diff_summaries', {})
        upd_frames = ref_res.get('updated_pre_frames', {})
        for seg in all_segment_data:
            cid = seg['id']
            if cid in ocr_map: seg['ocr_text'] = ocr_map[cid]
            if cid in det_map: seg['det_text'] = det_map[cid]
            if cid in diff_map: seg['diff_summary'] = diff_map[cid]
            if cid in upd_frames and isinstance(upd_frames[cid], list):
                seg['frames_with_ts'] = [(p, float(ts)) for p, ts in upd_frames[cid] if p]
    else:
        print('[Info] skip refine execution.')

    # Step 3: refine 过的段重新 caption
    try:
        plan_used = refinement_plan_with_results or refinement_plan or {}
        refined_ids = {t.get('clip_id') for t in (plan_used.get('targets') or [])}
        target_frame_map = {}
        for t in (plan_used.get('targets') or []):
            cid = t.get('clip_id'); rp = t.get('refinement_params') or {}
            val = rp.get('new_sampling_rate_per_30s') or rp.get('num_frames')
            if cid and val:
                try: target_frame_map[cid] = int(val)
                except: pass
        for seg in all_segment_data:
            if seg['id'] not in refined_ids: continue
            frames_all = [p for p,_ in seg.get('frames_with_ts') or []]
            if len(frames_all) <= 5: continue
            tgt_n = target_frame_map.get(seg['id'], len(frames_all))
            if tgt_n < len(frames_all):
                try:
                    import numpy as np
                    idxs = np.linspace(0, len(frames_all)-1, tgt_n, endpoint=True).astype(int).tolist()
                    sel = [frames_all[i] for i in idxs]
                except Exception:
                    sel = frames_all[:tgt_n]
            else:
                sel = frames_all
            seg['segment_caption'] = await generate_segment_caption(sel, llm_cfg)
            seg['caption_frame_count'] = len(sel)
            seg['caption_source'] = 'refined'
    except Exception as e:
        print(f'[CaptionRefine][WARN] {e}')

    # Step 4: 最终汇总 & 调用模型
    try:
        print('\n[CaptionSummary]')
        for seg in sorted(all_segment_data, key=lambda x: x['id']):
            print(f"[Cap][{seg['id']}] src={seg.get('caption_source')} frames={seg.get('caption_frame_count')} chars={len(seg.get('segment_caption') or '')}")
    except Exception:
        pass

    # Step 4.x: 最终段顺序整理（保证 邻居(prev, base, next) 顺序，如 _0 _1 _2 优先）
    final_segments = all_segment_data
    try:
        if neighbor_base_id:
            prev_id, next_id = None, None
            _nlist = _neighbor_ids(neighbor_base_id)
            if _nlist:
                if len(_nlist) >= 1:
                    prev_id = _nlist[0]
                if len(_nlist) >= 2:
                    next_id = _nlist[1]
            id_map = {s['id']: s for s in all_segment_data}
            ordered_ids = []
            if prev_id in id_map: ordered_ids.append(prev_id)
            if neighbor_base_id in id_map: ordered_ids.append(neighbor_base_id)
            if next_id in id_map: ordered_ids.append(next_id)
            remaining = [s for s in all_segment_data if s['id'] not in set(ordered_ids)]
            # 其余按开始时间排序，避免打乱时间线
            remaining_sorted = sorted(remaining, key=lambda x: (x.get('start_time', 0), x['id']))
            final_segments = [id_map[i] for i in ordered_ids] + remaining_sorted
    except Exception as e:
        print(f"[Order][WARN] {e}; fallback to existing order")

    # 如果 base_mode，则构建 images_base64，包含每段最多5帧，总计最多25张图片的 base64 字符串
    images_base64 = None
    if base_mode:
        imgs = []
        try:
            for seg in final_segments:
                seg_id = seg.get('id')
                frames_with_ts = seg.get('frames_with_ts') or []
                # 取前 5 帧
                frames = [(p, ts) for p, ts in frames_with_ts][:5]
                for idx, (p, ts) in enumerate(frames):
                    try:
                        b64 = image_to_base64(p)
                        if not b64:
                            continue
                        imgs.append({
                            'segment_id': seg_id,
                            'frame_index': idx,
                            'timestamp': float(ts),
                            'image_base64': b64,
                            'path': p
                        })
                    except Exception:
                        continue
        except Exception:
            imgs = []
        images_base64 = imgs[:25]

    # 如果 images_base64 是 dict list，则传递给 builder（builder 会原样转发给 LLM）
    answer_obj = await build_and_call_llm(query, final_segments, llm_cfg, base_mode=base_mode, images_base64=images_base64)
    return answer_obj

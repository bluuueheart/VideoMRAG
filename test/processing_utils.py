import os
import glob
import shutil
import subprocess
from typing import List, Tuple

import numpy as np
from PIL import Image
import math

from faster_whisper import WhisperModel  # 保持与原文件一致
from test_media_utils import (
    get_video_duration,
    get_video_resolution,
)


def _workdir_base() -> str:
    """
    返回仓库同级的 `workdir` 目录路径（与 `videorag` 同级）。
    优先使用环境变量 `WORKDIR`，否则在当前文件夹上溯两级寻找 workspace 根并创建 `workdir`。
    """
    env = os.environ.get("WORKDIR")
    if env:
        os.makedirs(env, exist_ok=True)
        return env

    # 当前文件位于 test/ 目录下，workdir 放在上两级（项目根）下
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.abspath(os.path.join(this_dir, ".."))
    workdir = os.path.join(repo_root, "workdir")
    os.makedirs(workdir, exist_ok=True)
    return workdir


def _segment_cache_dir(video_path: str, start_time: float, end_time: float) -> str:
    """为单个片段返回缓存目录路径。"""
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    seg_ms_start = int(round(float(start_time) * 1000))
    seg_ms_end = int(round(float(end_time) * 1000))
    base = _workdir_base()
    cache_dir = os.path.join(base, f"{video_id}_{seg_ms_start}_{seg_ms_end}")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _frame_cache_dir_for_segment(video_path: str, output_dir: str, start_time: float, end_time: float) -> str:
    """
    为单个视频片段生成独立的帧缓存目录，避免不同片段互相污染。
    使用毫秒时间戳以保证唯一性。
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    def _ms(t) -> str:
        try:
            return f"{int(round(float(t) * 1000))}"
        except Exception:
            return "na"
    return os.path.join(output_dir, f"{video_id}_{_ms(start_time)}_{_ms(end_time)}_frames")


def extract_frames_and_compress(
    video_path: str, output_dir: str, start_time: float = 0.0, end_time: float | None = None, num_frames: int = 5, target_height: int | None = None,
    show_compress_log: bool = False,
    existing_frames: list[tuple[str, float]] | None = None,
    force_size: tuple[int, int] | None = None
) -> tuple[list[tuple[str, float]], float | None]:
    """
    Extracts frames, resizes them, and returns paths with timestamps.
    If existing_frames are provided, it calculates the new timestamps,
    extracts only the missing frames, and merges them with existing ones.
    """
    # 归一化 end_time（若为 None，则根据视频时长或 30s 默认）
    if end_time is None:
        vid_dur = get_video_duration(video_path)
        start_time = 0.0 if start_time is None else float(start_time)
        end_time = min(30.0, vid_dur) if vid_dur else 30.0

    frame_dir = _frame_cache_dir_for_segment(video_path, output_dir, float(start_time), float(end_time))
    os.makedirs(frame_dir, exist_ok=True)

    # --- 尝试从全局 workdir 缓存加载原始前5帧（未压缩 BMP）和字幕文本 ---
    try:
        cache_dir = _segment_cache_dir(video_path, start_time, end_time)
    except Exception:
        cache_dir = None

    if cache_dir:
        cached_frames = sorted(glob.glob(os.path.join(cache_dir, "frame_*.bmp")))
        if cached_frames:
            # 使用缓存（按时间戳排序），但保持最多 num_frames
            use_frames = cached_frames[:num_frames]
            # 将 BMP 转为 JPEG 并按时间戳命名到本段 frame_dir
            existing_ms = []
            for src in use_frames:
                bn = os.path.basename(src)
                try:
                    ms_part = bn.split('_', 1)[1].split('.')[0]
                    ms = int(ms_part)
                except Exception:
                    # 后备使用索引毫秒（不太可能）
                    ms = None
                if ms is None:
                    dest_name = f"frame_{len(existing_ms)+1:04d}.jpg"
                else:
                    dest_name = f"frame_{ms:08d}.jpg"
                dest_path = os.path.join(frame_dir, dest_name)
                if not os.path.exists(dest_path):
                    try:
                        with Image.open(src) as im:
                            rgb = im.convert("RGB")
                            rgb.save(dest_path, format="JPEG", quality=95)
                    except Exception:
                        try:
                            shutil.copy2(src, dest_path)
                        except Exception:
                            pass
                if ms is not None:
                    existing_ms.append(ms/1000.0)

            # 计算需要的目标时间戳列表
            all_target_timestamps = np.linspace(start_time, end_time, num_frames, endpoint=False).tolist()
            # 计算已存在的时间戳集合（四舍五入到 ms 精度）
            existing_ts_set = {round(t, 3) for t in existing_ms}
            # 找出缺失的时间戳
            missing_ts = [ts for ts in all_target_timestamps if round(ts, 3) not in existing_ts_set]

            # 若没有缺失则直接返回缓存帧
            if not missing_ts:
                final_frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
                final_timestamps = np.linspace(start_time, end_time, len(final_frame_paths), endpoint=False).tolist()
                frames_with_ts = list(zip(final_frame_paths, final_timestamps))
                # 计算压缩比
                original_res = get_video_resolution(video_path)
                ratio = None
                if original_res and final_frame_paths:
                    try:
                        with Image.open(final_frame_paths[0]) as img:
                            new_res = img.size
                        original_pixels = original_res[0] * original_res[1]
                        new_pixels = new_res[0] * new_res[1]
                        if original_pixels > 0:
                            ratio = new_pixels / original_pixels
                    except Exception:
                        ratio = None

                print(f"[Cache] Loaded {len(frames_with_ts)} frames from cache for segment {os.path.basename(cache_dir)}")
                return frames_with_ts, ratio

            # 若存在缺失帧：使用 ffmpeg 在临时目录提取一组帧，然后选取最接近缺失时间戳的帧
            print(f"[Cache] Found {len(existing_ms)} cached frames, need to extract {len(missing_ts)} missing frames...")
            temp_frame_dir = os.path.join(frame_dir, "temp_new_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)
            # 设置较高 FPS 以保证包含接近目标时间点的帧
            required_fps = max( math.ceil((num_frames / (end_time - start_time)) * 2), 1) if (end_time - start_time) > 0 else num_frames
            # 依据 force_size/target_height 决定缩放
            if force_size:
                fw, fh = int(force_size[0]), int(force_size[1])
                scale_expr = f"scale={fw}:{fh}:flags=lanczos"
                vf_arg = f"fps={required_fps},{scale_expr}"
            elif target_height:
                scale_expr = f"scale=-1:{target_height}"
                vf_arg = f"fps={required_fps},{scale_expr}"
            else:
                vf_arg = f"fps={required_fps}"

            cmd_extract = [
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path,
                "-vf", vf_arg,
                os.path.join(temp_frame_dir, "tempframe_%04d.jpg")
            ]
            try:
                subprocess.run(cmd_extract, check=True, capture_output=True, text=True)
                temp_frames = sorted(glob.glob(os.path.join(temp_frame_dir, "*.jpg")))
                if temp_frames:
                    temp_timestamps = np.linspace(start_time, end_time, len(temp_frames), endpoint=False)
                    for ts_target in missing_ts:
                        best_match_idx = int(np.argmin(np.abs(temp_timestamps - ts_target)))
                        src_path = temp_frames[best_match_idx]
                        ts_ms = int(round(ts_target * 1000))
                        dest_filename = f"frame_{ts_ms:08d}.jpg"
                        dest_path = os.path.join(frame_dir, dest_filename)
                        if not os.path.exists(dest_path):
                            try:
                                shutil.copy2(src_path, dest_path)
                            except Exception:
                                try:
                                    os.replace(src_path, dest_path)
                                except Exception:
                                    pass
                        # also write to cache as BMP (if not exists)
                        try:
                            cache_bmp = os.path.join(cache_dir, f"frame_{ts_ms:08d}.bmp")
                            if not os.path.exists(cache_bmp):
                                with Image.open(dest_path) as im:
                                    im.convert("RGB").save(cache_bmp, format="BMP")
                        except Exception:
                            pass
                shutil.rmtree(temp_frame_dir)
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during cache-based incremental extraction: {e.stderr}")

            # 最终返回合并后的帧列表
            final_frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
            final_timestamps = np.linspace(start_time, end_time, len(final_frame_paths), endpoint=False).tolist()
            frames_with_ts = list(zip(final_frame_paths, final_timestamps))
            original_res = get_video_resolution(video_path)
            ratio = None
            if original_res and final_frame_paths:
                try:
                    with Image.open(final_frame_paths[0]) as img:
                        new_res = img.size
                    original_pixels = original_res[0] * original_res[1]
                    new_pixels = new_res[0] * new_res[1]
                    if original_pixels > 0:
                        ratio = new_pixels / original_pixels
                except Exception:
                    ratio = None

            print(f"[Cache] After supplementation loaded {len(frames_with_ts)} frames for segment {os.path.basename(cache_dir)}")
            return frames_with_ts, ratio

    duration = float(end_time) - float(start_time)
    if duration <= 0:
        print(f"[Frames] Invalid duration ({duration}) for segment. Skipping frame extraction.")
        return [], None

    all_target_timestamps = np.linspace(start_time, end_time, num_frames, endpoint=False).tolist()
    
    # --- 新增逻辑：基于现有帧进行增量提取 ---
    if existing_frames:
        print(f"[Frames] Augmenting {len(existing_frames)} existing frames to {num_frames} total.")
        existing_timestamps = {round(ts, 5) for _, ts in existing_frames}
        
        # 找出需要新提取的帧的时间戳
        new_timestamps_to_extract = []
        for ts in all_target_timestamps:
            if round(ts, 5) not in existing_timestamps:
                new_timestamps_to_extract.append(ts)

        if new_timestamps_to_extract:
            print(f"[Frames] Extracting {len(new_timestamps_to_extract)} new frames...")
            # 使用 FFmpeg 的 select filter 精确提取指定时间戳的帧
            # 注意：时间戳需要相对于视频开头，而不是片段开头
            select_expr = "+".join([f'eq(n,{int(round((ts - start_time) * (num_frames/duration)))})' for ts in new_timestamps_to_extract])
            
            # 为了避免文件名冲突和排序问题，我们将新帧保存到临时目录
            temp_frame_dir = os.path.join(frame_dir, "temp_new_frames")
            os.makedirs(temp_frame_dir, exist_ok=True)

            # 使用更可靠的fps和select过滤器组合
            # 计算一个足够高的FPS以确保能捕捉到所有时间点
            required_fps = (num_frames / duration) * 2 
            
            # 生成 select 表达式，选择最接近目标时间戳的帧
            # 'select' filter is 1-based index, pts_time is in seconds
            select_filter_parts = []
            for ts in new_timestamps_to_extract:
                 # We need to find the frame whose presentation time is closest to ts
                 # This is complex with select, so we extract at high rate and then pick
                 pass # We will select frames after extraction

            # 如果提供 force_size (width, height)，则使用精确缩放以获得统一尺寸
            # decide whether to include a scale filter
            if force_size:
                fw, fh = int(force_size[0]), int(force_size[1])
                scale_expr = f"scale={fw}:{fh}:flags=lanczos"
                vf_arg = f"fps={required_fps},{scale_expr}"
            elif target_height:
                scale_expr = f"scale=-1:{target_height}"
                vf_arg = f"fps={required_fps},{scale_expr}"
            else:
                # preserve original resolution when no scaling requested
                vf_arg = f"fps={required_fps}"

            cmd_extract = [
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path,
                "-vf", vf_arg,
                os.path.join(temp_frame_dir, "tempframe_%04d.jpg")
            ]
            try:
                subprocess.run(cmd_extract, check=True, capture_output=True, text=True)
                
                # 从临时目录中挑选最接近目标时间戳的帧
                temp_frames = sorted(glob.glob(os.path.join(temp_frame_dir, "*.jpg")))
                temp_timestamps = np.linspace(start_time, end_time, len(temp_frames), endpoint=False)

                for ts_target in new_timestamps_to_extract:
                    # 找到时间最接近的已提取帧
                    best_match_idx = np.argmin(np.abs(temp_timestamps - ts_target))
                    src_path = temp_frames[best_match_idx]
                    # 用目标时间戳重命名并移动到主帧目录
                    # 使用毫秒级精度命名以保证唯一性和排序
                    dest_filename = f"frame_{int(ts_target * 1000):08d}.jpg"
                    dest_path = os.path.join(frame_dir, dest_filename)
                    if not os.path.exists(dest_path):
                        # 使用复制而非移动，避免同一临时帧被多次选中后源文件缺失
                        try:
                            shutil.copy2(src_path, dest_path)
                        except Exception as copy_err:
                            # 若复制失败，尝试回退为移动（少数平台权限限制）
                            try:
                                os.replace(src_path, dest_path)
                            except Exception:
                                print(f"[Frames] Failed to materialize frame for ts={ts_target}: {copy_err}")

                # 清理临时目录
                shutil.rmtree(temp_frame_dir)

            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during incremental frame extraction: {e.stderr}")
        else:
            print("[Frames] No new frames needed.")

    # --- 如果没有现有帧，执行原始逻辑 ---
    else:
        # 检查是否已满足所需帧数
        existing_frame_files = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
        if len(existing_frame_files) == num_frames:
            print(f"[Frames] Frames already extracted for segment: {os.path.basename(frame_dir)}")
        else:
            # 清理旧帧，因为参数可能已变
            for f in existing_frame_files:
                os.remove(f)
            
            print(f"[Frames] Extracting {num_frames} frames for segment...")
            fps = num_frames / duration
            # 输出时根据 force_size 决定是否使用固定尺寸
            # decide whether to include a scale filter
            if force_size:
                fw, fh = int(force_size[0]), int(force_size[1])
                scale_expr = f"scale={fw}:{fh}:flags=lanczos"
                vf_arg = f"fps={fps},{scale_expr}"
            elif target_height:
                scale_expr = f"scale=-1:{target_height}"
                vf_arg = f"fps={fps},{scale_expr}"
            else:
                vf_arg = f"fps={fps}"

            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path,
                "-vf", vf_arg,
                os.path.join(frame_dir, "frame_%04d.jpg")
            ]
            try:
                subprocess.run(cmd, check=True, capture_output=True, text=True)
            except subprocess.CalledProcessError as e:
                print(f"[FFmpeg] Error during frame extraction: {e.stderr}")
    
    final_frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.jpg")))
    
    # 基于最终帧数重新计算精确时间戳
    final_timestamps = np.linspace(start_time, end_time, len(final_frame_paths), endpoint=False).tolist()
    frames_with_ts = list(zip(final_frame_paths, final_timestamps))

    # 计算像素压缩比
    original_res = get_video_resolution(video_path)
    ratio = None
    if original_res and final_frame_paths:
        with Image.open(final_frame_paths[0]) as img:
            new_res = img.size
        original_pixels = original_res[0] * original_res[1]
        new_pixels = new_res[0] * new_res[1]
        if original_pixels > 0:
            ratio = new_pixels / original_pixels

    # --- 将前最多 num_cache_save 张未压缩原始帧写入全局 workdir 缓存（BMP 格式） ---
    try:
        cache_dir = _segment_cache_dir(video_path, start_time, end_time)
    except Exception:
        cache_dir = None

    if cache_dir:
        try:
            # 只缓存前5张（或 num_frames 小于5时相应数量）
            num_cache_save = min(5, len(final_frame_paths))
            for i in range(num_cache_save):
                src = final_frame_paths[i]
                # 以毫秒命名保持唯一性
                ts = int(round(final_timestamps[i] * 1000))
                dest_bmp = os.path.join(cache_dir, f"frame_{ts:08d}.bmp")
                if not os.path.exists(dest_bmp):
                    try:
                        with Image.open(src) as im:
                            im.convert("RGB").save(dest_bmp, format="BMP")
                    except Exception:
                        try:
                            shutil.copy2(src, dest_bmp)
                        except Exception:
                            pass
        except Exception:
            pass

    return frames_with_ts, ratio


def transcribe_segment_audio(video_path: str, asr_model: WhisperModel, start_time: float, end_time: float) -> str:
    """Extracts the specific 30s audio segment and transcribes it using faster-whisper."""
    # 归一化 end_time（若为 None，则根据视频时长或 30s 默认）
    if end_time is None:
        vid_dur = get_video_duration(video_path)
        start_time = 0.0 if start_time is None else float(start_time)
        end_time = min(30.0, vid_dur) if vid_dur else 30.0

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    seg_ms_start = int(round(float(start_time) * 1000))
    seg_ms_end = int(round(float(end_time) * 1000))
    audio_path = os.path.join(os.path.dirname(video_path), f"{video_id}_{seg_ms_start}_{seg_ms_end}.mp3")

    # transcript cache path in workdir
    try:
        cache_dir = _segment_cache_dir(video_path, start_time, end_time)
    except Exception:
        cache_dir = None
    transcript_cache_path = os.path.join(cache_dir, "transcript.txt") if cache_dir else None

    # 如果 cache 中已有 transcript，则直接返回
    if transcript_cache_path and os.path.exists(transcript_cache_path):
        try:
            with open(transcript_cache_path, "r", encoding="utf-8") as f:
                txt = f.read().strip()
            if txt:
                print(f"[Cache] Loaded transcript from {transcript_cache_path}")
                return txt
        except Exception:
            pass

    if not os.path.exists(audio_path):
        try:
            cmd = ["ffmpeg", "-y", "-ss", str(start_time), "-to", str(end_time), "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
            print(f"[FFmpeg] Extracting audio segment {video_id} [{start_time}-{end_time}]...")
            subprocess.run(cmd, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"[FFmpeg] Error extracting audio for segment {video_path}: {e.stderr.decode()}")
            return ""

    try:
        print(f"[ASR] Transcribing segment {video_id} [{start_time}-{end_time}]...")
        segments, _ = asr_model.transcribe(audio_path, beam_size=5)
        transcript = " ".join([seg.text for seg in segments]).strip()

        # 写入缓存文件（如果可用）
        if transcript_cache_path:
            try:
                with open(transcript_cache_path, "w", encoding="utf-8") as f:
                    f.write(transcript)
            except Exception:
                pass

        return transcript
    except Exception as e:
        print(f"[ASR] Error during transcription for {audio_path}: {e}")
        return ""

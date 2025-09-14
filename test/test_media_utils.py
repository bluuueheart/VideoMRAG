import os
import subprocess
import urllib
import shutil
import glob
import time
from PIL import Image
import numpy as np


def download_file(url: str, target_dir: str) -> str:
    """Downloads a file from a URL to a target directory."""
    import urllib.request
    import urllib.parse
    filename = os.path.basename(urllib.parse.unquote(url).split('?')[0])
    filepath = os.path.join(target_dir, filename)
    if os.path.exists(filepath):
        print(f"[Download] File already exists: {filepath}")
        return filepath
    print(f"[Download] Downloading {url} to {filepath}")
    # Try a few times for transient network issues
    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            # Use urlopen with a timeout to get clearer exceptions
            with urllib.request.urlopen(url, timeout=15) as resp:
                # Stream to file
                with open(filepath, 'wb') as f:
                    shutil.copyfileobj(resp, f)
            return filepath
        except Exception as e:
            # Provide more actionable message for DNS/network errors
            msg = str(e)
            if 'Name or service not known' in msg or 'Temporary failure in name resolution' in msg:
                print(f"[Download][Attempt {attempt}/{max_attempts}] DNS/network error for {url}: {e}")
            else:
                print(f"[Download][Attempt {attempt}/{max_attempts}] Error downloading {url}: {e}")
            # On last attempt, return empty string
            if attempt == max_attempts:
                print(f"[Download] Failed after {max_attempts} attempts: {url}")
                return ""
            # small backoff
            try:
                time.sleep(2 * attempt)
            except Exception:
                pass


def get_video_resolution(video_path: str) -> tuple[int, int] | None:
    """Gets video resolution using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=width,height", "-of", "csv=s=x:p=0", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        width, height = map(int, result.stdout.strip().split('x'))
        return width, height
    except Exception as e:
        print(f"[FFprobe] Error getting resolution for {video_path}: {e}")
        return None


def get_video_duration(video_path: str) -> float | None:
    """Gets video duration in seconds using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"[FFprobe] Error getting duration for {video_path}: {e}")
        return None


def is_valid_video(video_path: str) -> bool:
    dur = get_video_duration(video_path)
    return dur is not None and dur > 0


def repair_mp4_faststart(src_path: str) -> str | None:
    """Try to repair an MP4 by remuxing with faststart. Returns repaired path or None."""
    try:
        if not os.path.exists(src_path):
            return None
        repaired = os.path.splitext(src_path)[0] + "_fixed.mp4"
        cmd = [
            "ffmpeg", "-y", "-i", src_path,
            "-c", "copy", "-movflags", "+faststart",
            repaired
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return repaired if os.path.exists(repaired) else None
    except Exception as e:
        print(f"[FFmpeg] Repair failed for {src_path}: {e}")
        return None


def ensure_valid_video_or_skip(url: str, work_dir: str, video_path: str) -> str | None:
    """Ensure the MP4 at video_path is valid. If invalid, force redownload; if still invalid, try faststart repair. Returns a valid path or None to skip."""
    if is_valid_video(video_path):
        return video_path
    print(f"[Validate] Detected invalid video file: {video_path}. Forcing re-download...")
    try:
        if os.path.exists(video_path):
            os.remove(video_path)
    except Exception:
        pass
    new_path = download_file(url, work_dir)
    if new_path and is_valid_video(new_path):
        return new_path
    print(f"[Validate] Re-download did not produce a valid MP4. Attempting repair...")
    repaired = repair_mp4_faststart(new_path or video_path)
    if repaired and is_valid_video(repaired):
        # Replace original path reference with repaired file
        try:
            shutil.move(repaired, new_path or video_path)
            final_path = new_path or video_path
        except Exception:
            final_path = repaired
        print(f"[Validate] Repair succeeded: {final_path}")
        return final_path if is_valid_video(final_path) else None
    print(f"[Validate] Repair failed. Skipping this segment: {url}")
    return None


def image_compression_ratio(video_path: str, frame_paths: list[str]) -> float | None:
    original_res = get_video_resolution(video_path)
    if not original_res or not frame_paths:
        return None
    with Image.open(frame_paths[0]) as img:
        new_res = img.size
    original_pixels = original_res[0] * original_res[1]
    new_pixels = new_res[0] * new_res[1]
    if original_pixels > 0:
        return new_pixels / original_pixels
    return None

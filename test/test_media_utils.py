import os
import subprocess
import urllib
import shutil
import glob
import time
from PIL import Image
import numpy as np
import shutil
# from shutil import which
try:
    # optional dependency that may provide a bundled ffmpeg binary
    import imageio_ffmpeg
except Exception:
    imageio_ffmpeg = None


def get_imageio_ffmpeg_exe() -> str | None:
    """Return a usable ffmpeg executable path from imageio_ffmpeg in a
    backward/forward-compatible way.

    Some versions expose `get_exe()`, others `get_ffmpeg_exe()`. Try
    known callables and return the first working result, otherwise None.
    """
    if imageio_ffmpeg is None:
        return None
    # try common names in order
    candidates = [
        getattr(imageio_ffmpeg, 'get_exe', None),
        getattr(imageio_ffmpeg, 'get_ffmpeg_exe', None),
        getattr(imageio_ffmpeg, 'get_ffmpeg_exe', None),
    ]
    for fn in candidates:
        if callable(fn):
            try:
                path = fn()
                if path:
                    return path
            except Exception:
                # ignore and try next
                continue
    return None
import re


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
        ffmpeg_path = get_imageio_ffmpeg_exe()
        if not ffmpeg_path:
            raise FileNotFoundError("imageio-ffmpeg did not provide ffmpeg executable")
        # run ffmpeg -i and parse stderr for resolution
        cmd = [ffmpeg_path, "-i", video_path]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stderr = proc.stderr or proc.stdout or ""
        m = re.search(r"(\d{2,5})x(\d{2,5})", stderr)
        if m:
            width, height = int(m.group(1)), int(m.group(2))
            return width, height
        raise RuntimeError("Unable to parse resolution from ffmpeg output")
    except Exception as e:
        print(f"[FFmpeg] Error getting resolution for {video_path}: {e}")
        return None


def get_video_duration(video_path: str) -> float | None:
    """Gets video duration in seconds using ffprobe."""
    try:
        ffmpeg_path = get_imageio_ffmpeg_exe()
        if not ffmpeg_path:
            raise FileNotFoundError("imageio-ffmpeg did not provide ffmpeg executable")
        proc = subprocess.run([ffmpeg_path, "-i", video_path], capture_output=True, text=True)
        stderr = proc.stderr or proc.stdout or ""
        m = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)", stderr)
        if m:
            h = float(m.group(1))
            mm = float(m.group(2))
            s = float(m.group(3))
            return h * 3600.0 + mm * 60.0 + s
        raise RuntimeError("Unable to parse duration from ffmpeg output")
    except Exception as e:
        print(f"[FFmpeg] Error getting duration for {video_path}: {e}")
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
        ffmpeg_path = get_imageio_ffmpeg_exe()
        if not ffmpeg_path:
            raise FileNotFoundError("imageio-ffmpeg did not provide ffmpeg executable")
        cmd = [
            ffmpeg_path, "-y", "-i", src_path,
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
    print(f"[Validate] Detected invalid video file: {video_path}. Network re-download disabled in local-only mode.")
    # Attempt local repair via ffmpeg faststart remux only. Do NOT attempt network download.
    repaired = repair_mp4_faststart(video_path)
    if repaired and is_valid_video(repaired):
        # Replace original path reference with repaired file
        try:
            shutil.move(repaired, video_path)
            final_path = video_path
        except Exception:
            final_path = repaired
        print(f"[Validate] Repair succeeded: {final_path}")
        return final_path if is_valid_video(final_path) else None
    print(f"[Validate] Repair failed or not possible. Skipping this segment: {video_path}")
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

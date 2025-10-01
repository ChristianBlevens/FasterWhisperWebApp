"""
Stage 5: Clip Downloading with CFR Normalization
Downloads and normalizes video clips to prevent concatenation glitches
"""
import os
import yt_dlp
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0.0


def download_video_clip(
    video_url: str,
    start_time: float,
    end_time: float,
    output_path: str
) -> Optional[str]:
    """
    Download and normalize video clip to CFR

    Downloads clip with forced keyframes at cuts, then re-encodes to:
    - 30 FPS constant framerate
    - H.264 video codec
    - AAC audio codec
    - No B-frames to prevent frame reordering

    Args:
        video_url: YouTube video URL
        start_time: Start time in seconds
        end_time: End time in seconds
        output_path: Output file path

    Returns:
        Path to normalized clip or None if failed
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    temp_output = output_path.replace('.mp4', '.temp.mp4')

    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        'outtmpl': temp_output,
        'quiet': True,
        'no_warnings': True,
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_time, end_time)]),
        'force_keyframes_at_cuts': True,
        'format_sort': ['codec:h264', 'codec:aac'],
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

    try:
        # Download clip with forced keyframes at cuts
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([video_url])

        if not os.path.exists(temp_output) or os.path.getsize(temp_output) < 1000:
            return None

        # Re-encode to normalize framerate and codec
        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_output,
            '-vf', 'fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-bf', '0',  # No B-frames - prevents PTS/DTS reordering issues
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            output_path
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        # Clean up temp file
        if os.path.exists(temp_output):
            os.remove(temp_output)

        if result.returncode != 0:
            print(f"FFmpeg normalization failed: {result.stderr[:200]}", flush=True)
            return None

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            return output_path
        else:
            return None

    except Exception as e:
        print(f"Error downloading clip: {e}", flush=True)
        if os.path.exists(temp_output):
            try:
                os.remove(temp_output)
            except:
                pass
        return None


def download_clips_batch(
    clips: List[Dict],
    clips_dir: Path,
    gb_limit: float = 10.0,
    max_workers: int = 32,
    progress_callback: Optional[Callable] = None
) -> Dict[str, any]:
    """
    Download video clips with GB limit and parallel processing

    Args:
        clips: List of clip definitions
        clips_dir: Output directory for clips
        gb_limit: Maximum GB to download before stopping
        max_workers: Maximum parallel downloads (default: 32)
        progress_callback: Optional callback(current, total, downloaded_gb)

    Returns:
        Dictionary with download results
    """
    start_time = time.time()
    clips_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    total_size_gb = 0.0

    pending_clips = [c for c in clips if not c.get('downloaded', False)]

    print(f"Stage 5: Downloading {len(pending_clips)} clips (limit: {gb_limit} GB)", flush=True)

    def download_single_clip(clip):
        """Download a single clip"""
        clip_id = clip['clip_id']
        filename = f"clip_{clip_id:05d}.mp4"
        output_path = clips_dir / filename

        # Skip if exists
        if output_path.exists():
            file_size = get_file_size_gb(str(output_path))
            clip['downloaded'] = True
            clip['clip_file'] = filename
            clip['file_size_gb'] = file_size
            return 'skipped', file_size, clip

        # Download
        result = download_video_clip(
            clip['video_url'],
            clip['absolute_start'],
            clip['absolute_end'],
            str(output_path)
        )

        if result:
            file_size = get_file_size_gb(result)
            clip['downloaded'] = True
            clip['clip_file'] = filename
            clip['file_size_gb'] = file_size
            return 'success', file_size, clip
        else:
            return 'failed', 0.0, clip

    # Process clips with size limit
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []

        for clip in pending_clips:
            if total_size_gb >= gb_limit:
                break

            future = executor.submit(download_single_clip, clip)
            futures.append(future)

        for future in as_completed(futures):
            status, file_size, clip = future.result()

            if status == 'success':
                downloaded_count += 1
                total_size_gb += file_size
                print(f"Downloaded clip {clip['clip_id']}: {clip['word']} ({file_size:.3f} GB)", flush=True)
            elif status == 'skipped':
                skipped_count += 1
                total_size_gb += file_size
            else:
                failed_count += 1
                print(f"Failed clip {clip['clip_id']}", flush=True)

            if total_size_gb >= gb_limit:
                print(f"GB limit reached: {total_size_gb:.2f} GB", flush=True)
                for f in futures:
                    f.cancel()
                break

            # Progress callback
            if progress_callback:
                progress_callback(
                    downloaded_count + skipped_count + failed_count,
                    len(pending_clips),
                    total_size_gb
                )

    elapsed_time = time.time() - start_time
    result = {
        'downloaded': downloaded_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'total_size_gb': total_size_gb,
        'limit_reached': total_size_gb >= gb_limit,
        'pending_count': len([c for c in clips if not c.get('downloaded', False)])
    }

    print(f"\nStage 5 complete: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed ({total_size_gb:.2f} GB, {elapsed_time / 60:.1f} min)", flush=True)

    return result
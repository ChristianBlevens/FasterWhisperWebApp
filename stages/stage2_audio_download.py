"""
Stage 2: Full Audio Download with Anti-Throttling
Downloads complete video audio using yt-dlp with comprehensive anti-throttling measures
"""
import os
import yt_dlp
import subprocess
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Callable

download_active = False
last_downloaded_bytes = 0
current_download_path = None


def get_file_size_gb(file_path: str) -> float:
    """Get file size in GB"""
    if os.path.exists(file_path):
        return os.path.getsize(file_path) / (1024 ** 3)
    return 0.0


def get_anti_throttling_opts() -> Dict:
    """
    Returns yt-dlp options optimized for anti-throttling

    Uses aria2c multi-connection downloads and 10MB chunk size
    to avoid YouTube throttling mechanisms
    """
    return {
        'format': 'worstaudio/worst',
        'http_chunk_size': 10485760,  # 10MB chunk size
        'concurrent_fragment_downloads': 8,
        'external_downloader': 'aria2c',
        'external_downloader_args': [
            '-c',
            '-j', '3',
            '-x', '16',
            '-s', '16',
            '-k', '1M',
            '--max-tries', '10',
            '--retry-wait', '3',
            '--connect-timeout', '60',
            '--timeout', '60',
        ],
        'extractor_args': {
            'youtube': {
                'player_client': ['mweb', 'ios', 'android', 'web'],
                'skip': ['dash', 'hls'],
            }
        },
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-us,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Sec-Fetch-Mode': 'navigate',
        },
        'retries': 15,
        'fragment_retries': 15,
        'file_access_retries': 10,
        'socket_timeout': 60,
        'ignoreerrors': False,
        'no_warnings': True,
        'continuedl': True,
        'nopart': False,
        'quiet': True,
        'no_color': True,
        'noprogress': True,
        'verbose': False,
    }


def speed_monitor_thread(temp_file):
    """Monitor download speed in real-time"""
    global download_active, last_downloaded_bytes

    # aria2c creates a .part file during download
    part_file = temp_file + '.part'

    while download_active:
        # Check both .part file (during download) and final file
        check_file = part_file if os.path.exists(part_file) else temp_file

        if os.path.exists(check_file):
            current_size = os.path.getsize(check_file)
            speed_bytes = current_size - last_downloaded_bytes
            speed_mb = speed_bytes / (1024 * 1024)

            if speed_mb > 0:
                print(f"Download speed: {speed_mb:.2f} MB/s", flush=True)

            last_downloaded_bytes = current_size

        time.sleep(1)


def download_full_video_audio(
    video_url: str,
    output_path: str
) -> Optional[str]:
    """
    Download complete video audio and convert to 16kHz WAV

    Args:
        video_url: YouTube video URL
        output_path: Output file path for WAV

    Returns:
        Path to downloaded WAV file or None if failed
    """
    global download_active, last_downloaded_bytes, current_download_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temp_file = output_path.replace('.wav', '.temp.m4a')

    download_start_time = time.time()
    last_downloaded_bytes = 0
    current_download_path = temp_file

    def progress_hook(d):
        global download_active
        status = d.get('status', '')

        if status == 'downloading':
            download_active = True

        elif status == 'finished':
            download_active = False

        elif status == 'error':
            download_active = False

    ydl_opts = get_anti_throttling_opts()
    ydl_opts['outtmpl'] = temp_file
    ydl_opts['progress_hooks'] = [progress_hook]

    try:
        download_active = True

        monitor = threading.Thread(target=speed_monitor_thread, args=(temp_file,), daemon=True)
        monitor.start()

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            error_code = ydl.download([video_url])
            if error_code != 0:
                print(f"Download error code: {error_code}", flush=True)

        download_active = False
        monitor.join(timeout=2)

        if not os.path.exists(temp_file):
            print(f"Download failed: no output file", flush=True)
            return None

        temp_size = os.path.getsize(temp_file)
        if temp_size == 0:
            print(f"Download failed: empty file", flush=True)
            os.remove(temp_file)
            return None

        download_elapsed = time.time() - download_start_time
        download_speed_mb = (temp_size / (1024*1024)) / download_elapsed if download_elapsed > 0 else 0
        print(f"Downloaded: {temp_size / (1024*1024):.1f} MB ({download_speed_mb:.1f} MB/s, {download_elapsed:.1f}s)", flush=True)

        wav_path = output_path if output_path.endswith('.wav') else f"{output_path}.wav"

        ffmpeg_cmd = [
            'ffmpeg', '-i', temp_file,
            '-acodec', 'pcm_s16le', '-ar', '16000',
            wav_path, '-y', '-loglevel', 'error'
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if os.path.exists(temp_file):
            os.remove(temp_file)

        if result.returncode != 0:
            print(f"FFmpeg failed: {result.stderr if result.stderr else 'unknown error'}", flush=True)
            return None

        if not os.path.exists(wav_path) or os.path.getsize(wav_path) == 0:
            print(f"Conversion failed: invalid WAV file", flush=True)
            if os.path.exists(wav_path):
                os.remove(wav_path)
            return None

        return wav_path

    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}", flush=True)
        download_active = False
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        return None


def download_videos_batch(
    videos: List[Dict],
    output_dir: Path,
    gb_limit: float = 5.0,
    progress_callback: Optional[Callable] = None
) -> Dict[str, any]:
    """
    Download full video audio files with GB limit

    Args:
        videos: List of video definitions with 'video_url' and 'video_id'
        output_dir: Output directory for audio files
        gb_limit: Maximum GB to download before stopping
        progress_callback: Optional callback(current, total, downloaded_gb)

    Returns:
        Dictionary with download results
    """
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded_count = 0
    skipped_count = 0
    failed_count = 0
    total_size_gb = 0.0

    pending_videos = [v for v in videos if not v.get('downloaded', False)]

    print(f"Stage 2: Downloading {len(pending_videos)} videos (limit: {gb_limit} GB)", flush=True)

    for idx, video in enumerate(pending_videos):
        if total_size_gb >= gb_limit:
            print(f"GB limit reached: {total_size_gb:.2f} GB", flush=True)
            break

        video_id = video.get('video_id', idx)
        video_url = video.get('video_url')
        video_title = video.get('video_title', f'video_{video_id}')

        safe_title = "".join(c for c in video_title if c.isalnum() or c in (' ', '-', '_')).rstrip()
        filename = f"video_{video_id}_{safe_title[:50]}.wav"
        output_path = output_dir / filename

        if output_path.exists():
            file_size = get_file_size_gb(str(output_path))
            total_size_gb += file_size
            skipped_count += 1
            video['downloaded'] = True
            video['audio_file'] = filename
            continue

        if total_size_gb >= gb_limit:
            break

        print(f"\n[{idx + 1}/{len(pending_videos)}] {video_title}", flush=True)

        result = download_full_video_audio(video_url, str(output_path))

        if result:
            file_size = get_file_size_gb(result)
            total_size_gb += file_size
            downloaded_count += 1
            video['downloaded'] = True
            video['audio_file'] = filename
        else:
            failed_count += 1

        if progress_callback:
            progress_callback(idx + 1, len(pending_videos), total_size_gb)

        if idx < len(pending_videos) - 1:
            time.sleep(2)

    elapsed_time = time.time() - start_time
    result = {
        'downloaded': downloaded_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'total_size_gb': total_size_gb,
        'limit_reached': total_size_gb >= gb_limit,
        'pending_count': len([v for v in videos if not v.get('downloaded', False)])
    }

    print(f"\nStage 2 complete: {downloaded_count} downloaded, {skipped_count} skipped, {failed_count} failed ({total_size_gb:.2f} GB, {elapsed_time / 60:.1f} min)", flush=True)

    return result
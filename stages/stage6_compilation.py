"""
Stage 6: Final Compilation
Concatenates clips using FFmpeg concat filter with timestamp normalization
"""
import os
import subprocess
import time
from pathlib import Path
from typing import List, Dict, Optional


def compile_video(
    clips: List[Dict],
    clips_dir: Path,
    output_path: Path,
    max_length: Optional[float] = None
) -> Dict[str, any]:
    """
    Compile clips using FFmpeg concat filter

    Args:
        clips: List of clip definitions
        clips_dir: Directory containing clip files
        output_path: Output path for final compilation
        max_length: Optional maximum length in seconds

    Returns:
        Dictionary with compilation results
    """
    start_time = time.time()

    downloaded_clips = [c for c in clips if c.get('downloaded', False)]

    if not downloaded_clips:
        return {
            'success': False,
            'error': 'No downloaded clips found',
            'total_clips': 0,
            'final_duration': 0
        }

    print(f"Stage 6: Compiling {len(downloaded_clips)} clips", flush=True)

    # Check which clips have audio streams
    input_files = []
    has_audio = []
    failed_clips = []

    for clip in downloaded_clips:
        clip_file = clip.get('clip_file')
        if not clip_file:
            continue

        clip_path = clips_dir / clip_file
        if not clip_path.exists():
            failed_clips.append(clip_file)
            continue

        # Check if clip has audio stream
        probe_cmd = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries', 'stream=codec_type', '-of', 'csv=p=0', str(clip_path)]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        clip_has_audio = bool(probe_result.stdout.strip())

        input_files.append(str(clip_path))
        has_audio.append(clip_has_audio)

    # Build filter_complex with audio generation for clips without audio
    video_filters = []
    audio_filters = []

    for idx, (input_file, clip_has_audio) in enumerate(zip(input_files, has_audio)):
        # Reset timestamps to zero for each clip
        video_filters.append(f"[{idx}:v]setpts=PTS-STARTPTS[v{idx}]")

        if clip_has_audio:
            audio_filters.append(f"[{idx}:a]asetpts=PTS-STARTPTS[a{idx}]")
        else:
            # Generate silent audio for clips without audio stream
            audio_filters.append(f"[{idx}:v]anullsrc=channel_layout=stereo:sample_rate=44100[a{idx}]")

    if not input_files:
        return {
            'success': False,
            'error': 'No valid clips to compile',
            'total_clips': 0,
            'final_duration': 0,
            'failed_clips': failed_clips
        }

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build FFmpeg concat filter command
        n = len(input_files)
        filter_complex = ";".join(video_filters + audio_filters) + ";"
        # Interleave streams: [v0][a0][v1][a1]...
        filter_complex += "".join([f"[v{i}][a{i}]" for i in range(n)])
        filter_complex += f"concat=n={n}:v=1:a=1[outv][outa]"

        ffmpeg_cmd = ['ffmpeg']
        for input_file in input_files:
            ffmpeg_cmd.extend(['-i', input_file])

        ffmpeg_cmd.extend([
            '-filter_complex', filter_complex,
            '-map', '[outv]',
            '-map', '[outa]',
            '-c:v', 'libx264',
            '-preset', 'fast',
            '-crf', '23',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y',
            str(output_path)
        ])

        print(f"Running FFmpeg concatenation", flush=True)

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"FFmpeg stderr output:\n{result.stderr}", flush=True)
            print(f"\nFilter complex:\n{filter_complex}", flush=True)
            return {
                'success': False,
                'error': f'FFmpeg failed - check logs',
                'total_clips': 0,
                'final_duration': 0
            }

        if not os.path.exists(output_path) or os.path.getsize(output_path) < 1000:
            return {
                'success': False,
                'error': 'Output file not created or invalid',
                'total_clips': 0,
                'final_duration': 0
            }

        final_size_gb = os.path.getsize(output_path) / (1024 ** 3)

        # Get duration from ffprobe
        probe_cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', str(output_path)]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        total_duration = float(probe_result.stdout.strip()) if probe_result.returncode == 0 else 0.0

        elapsed_time = time.time() - start_time
        print(f"\nStage 6 complete: {len(input_files)} clips, {total_duration:.1f}s ({final_size_gb:.2f} GB, {elapsed_time / 60:.1f} min)", flush=True)

        return {
            'success': True,
            'output_path': str(output_path),
            'total_clips': len(input_files),
            'final_duration': total_duration,
            'file_size_gb': final_size_gb,
            'failed_clips': failed_clips
        }

    except Exception as e:
        print(f"Error during compilation: {e}", flush=True)
        return {
            'success': False,
            'error': str(e),
            'total_clips': 0,
            'final_duration': 0
        }
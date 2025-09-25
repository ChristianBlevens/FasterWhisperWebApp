"""
Video Compilation Module
Handles video processing, clip extraction, and compilation creation
"""

import os
import time
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yt_dlp
from moviepy.editor import VideoFileClip, concatenate_videoclips
import wave
import json


def calculate_segments(duration_seconds, segment_size=600):
    """
    Calculate processing segments for a video
    Videos ≤20 minutes: single segment
    Videos >20 minutes: 10-minute segments with overflow in last
    """
    if duration_seconds <= segment_size * 2:  # ≤20 minutes
        return [(0, duration_seconds)]

    segments = []
    for start in range(0, duration_seconds, segment_size):
        end = min(start + segment_size, duration_seconds)
        segments.append((start, end))

    return segments


def get_video_duration(url):
    """Get video duration without downloading"""
    ydl_opts = {'quiet': True, 'no_warnings': True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get('duration', 0), info.get('title', 'Unknown')
    except Exception as e:
        print(f"Error getting video info: {e}")
        return 0, 'Unknown'


def download_audio_segment(url, start_time, end_time, output_path):
    """
    Download only audio from specific segment of YouTube video
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': output_path.replace('.wav', ''),  # yt-dlp adds extension
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'external_downloader': 'ffmpeg',
        'external_downloader_args': {
            'ffmpeg': ['-ss', str(start_time), '-to', str(end_time)]
        }
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Check if file was created with .wav extension
        expected_file = output_path if output_path.endswith('.wav') else f"{output_path}.wav"
        if os.path.exists(expected_file):
            return expected_file

        # Sometimes yt-dlp creates files with different extensions
        base_path = output_path.replace('.wav', '')
        for ext in ['.wav', '.m4a', '.mp3', '.opus']:
            test_path = f"{base_path}{ext}"
            if os.path.exists(test_path):
                # Convert to WAV if needed
                if ext != '.wav':
                    wav_path = f"{base_path}.wav"
                    subprocess.run([
                        'ffmpeg', '-i', test_path, '-acodec', 'pcm_s16le',
                        '-ar', '16000', wav_path, '-y'
                    ], capture_output=True)
                    os.remove(test_path)
                    return wav_path
                return test_path

        print(f"Warning: Audio file not found after download")
        return None

    except Exception as e:
        print(f"Error downloading audio segment: {e}")
        return None


def download_video_clip(url, start_time, end_time, output_path, padding_before=1.0, padding_after=1.0):
    """
    Download specific video clip with audio using timestamps from transcript
    """
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Add padding to timestamps
    start_with_padding = max(0, start_time - padding_before)
    end_with_padding = end_time + padding_after
    duration = end_with_padding - start_with_padding

    # Use a more reliable approach with consistent video parameters
    temp_path = output_path + '.temp'

    # Use yt-dlp's full capabilities for best results
    ydl_opts = {
        'format': 'bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best',
        'outtmpl': output_path,
        'quiet': True,
        'no_warnings': True,
        # Use yt-dlp's built-in download sections feature for precise timing
        'download_ranges': yt_dlp.utils.download_range_func(None, [(start_with_padding, end_with_padding)]),
        # Use yt-dlp's postprocessors for consistent format and quality
        'postprocessors': [
            {
                'key': 'FFmpegVideoRemuxer',
                'preferedformat': 'mp4',
            },
            {
                'key': 'FFmpegVideoConvertor',
                'preferedformat': 'mp4',
            }
        ],
        # Ensure consistent encoding through yt-dlp
        'format_sort': ['codec:h264', 'codec:aac'],
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

    try:
        print(f"🎥 Downloading clip: {start_time:.1f}-{end_time:.1f}s")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
            print(f"✅ Downloaded clip: {output_path}")
            return output_path
        else:
            print(f"❌ Clip file not created or too small: {output_path}")
            return None

    except Exception as e:
        print(f"❌ Error downloading video clip: {e}")
        return None


def parse_transcript_for_targets(segments, target_words, merge_gap=2.0):
    """
    Parse transcript segments for target words
    Returns list of clip timestamps with merge optimization
    """
    found_clips = []
    target_words_lower = [w.lower().strip() for w in target_words]
    print(f"🎯 Target words: {target_words_lower}")  # Debug logging

    for segment in segments:
        if hasattr(segment, 'words') and segment.words:
            for word_obj in segment.words:
                word_text = word_obj.word.strip().lower()
                # Remove punctuation for matching
                word_clean = ''.join(c for c in word_text if c.isalnum() or c.isspace())

                # Check if word matches any target (precise matching only)
                for target in target_words_lower:
                    # More restrictive matching for names
                    if (word_clean == target or  # exact match
                        (len(word_clean) >= 5 and len(target) >= 5 and
                         (word_clean in target or target in word_clean)) or  # substring match for longer names only
                        (len(word_clean) >= 6 and len(target) >= 6 and
                         abs(len(word_clean) - len(target)) <= 2 and
                         word_clean[:4] == target[:4])):  # similar length + same prefix for name variations
                        print(f"✅ Match found: '{word_clean}' matches target '{target}'")  # Debug logging
                        clip_data = {
                            'word': word_obj.word,
                            'start': word_obj.start,
                            'end': word_obj.end,
                            'confidence': getattr(word_obj, 'probability', 1.0)
                        }
                        found_clips.append(clip_data)
                        break  # Don't duplicate if multiple targets match

    # Merge nearby clips
    merged_clips = []
    if found_clips:
        current_clip = found_clips[0].copy()

        for next_clip in found_clips[1:]:
            # Check if clips should be merged (gap between clips is within merge_gap)
            gap_between_clips = next_clip['start'] - current_clip['end']
            if merge_gap > 0 and gap_between_clips <= merge_gap and gap_between_clips >= -0.1:
                # Merge clips
                current_clip['end'] = next_clip['end']
                current_clip['word'] += f" ... {next_clip['word']}"
            else:
                # Save current and start new
                merged_clips.append(current_clip)
                current_clip = next_clip.copy()

        # Don't forget last clip
        merged_clips.append(current_clip)

    return merged_clips


def create_compilation_video(clips_info, output_path, max_length=None):
    """
    Merge video clips into compilation using moviepy
    """
    from moviepy.editor import VideoFileClip, concatenate_videoclips
    import os

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_clips = []
    total_duration = 0
    failed_clips = []

    for clip_info in clips_info:
        if max_length and total_duration >= max_length:
            print(f"⚠️ Reached max compilation length: {max_length}s")
            break

        clip_path = clip_info['path']
        if os.path.exists(clip_path):
            try:
                # Load video with better error handling
                video = VideoFileClip(clip_path)

                # Validate video has required attributes
                if not hasattr(video, 'duration') or video.duration <= 0:
                    print(f"❌ Invalid video duration for {clip_path}")
                    video.close()
                    failed_clips.append(clip_path)
                    continue

                # Check if adding this clip exceeds max_length
                if max_length and total_duration + video.duration > max_length:
                    # Trim the clip to fit
                    remaining_time = max_length - total_duration
                    if remaining_time > 0.1:  # Only trim if there's meaningful time left
                        video = video.subclip(0, remaining_time)
                        print(f"⚠️ Trimming last clip to fit within {max_length}s limit")
                    else:
                        video.close()
                        break  # No more room

                # Ensure video has consistent properties
                if video.fps is None:
                    video = video.set_fps(30)  # Set default fps

                video_clips.append(video)
                total_duration += video.duration
                print(f"✅ Added clip: {clip_info.get('word', 'unknown')} ({video.duration:.1f}s)")

            except Exception as e:
                print(f"❌ Error loading clip {clip_path}: {e}")
                failed_clips.append(clip_path)
                # Try to clean up the problematic file
                try:
                    os.remove(clip_path)
                    print(f"🗑️ Removed problematic clip: {clip_path}")
                except:
                    pass

    if video_clips:
        try:
            print(f"🎬 Concatenating {len(video_clips)} clips...")

            # Concatenate all clips
            final_video = concatenate_videoclips(video_clips, method="compose")

            print(f"💾 Writing final video to {output_path}...")
            # Write output with audio
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='/tmp/temp-audio.m4a',
                remove_temp=True,
                fps=30,
                logger=None  # Suppress moviepy's verbose output
            )

            # Clean up
            for video in video_clips:
                video.close()

            print(f"✅ Compilation created: {output_path} ({total_duration:.1f}s)")
            return output_path, total_duration

        except Exception as e:
            print(f"❌ Error creating compilation: {e}")
            # Clean up on error
            for video in video_clips:
                try:
                    video.close()
                except:
                    pass
            return None, 0
    else:
        print("⚠️ No valid clips to compile")
        return None, 0


def get_audio_duration(audio_path):
    """Get duration of audio file in seconds"""
    try:
        with wave.open(audio_path, 'rb') as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
            duration = frames / float(rate)
            return duration
    except Exception as e:
        print(f"Error getting audio duration: {e}")
        # Fallback to ffprobe
        try:
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries',
                 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                 audio_path],
                capture_output=True, text=True
            )
            return float(result.stdout.strip())
        except:
            return 0
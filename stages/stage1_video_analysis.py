"""
Stage 1: Video Analysis
Analyzes YouTube URLs and extracts video metadata
"""
import yt_dlp
from typing import List, Dict, Tuple, Any
import time


def extract_video_info(url: str) -> Tuple[List[str], str, int]:
    """
    Extract video URLs, title, and total duration from YouTube URL
    Supports single videos, playlists, and channels
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'lazy_playlist': True,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if 'entries' in info:
                entries = info['entries']
                urls = []
                total_duration = 0

                for entry in entries:
                    if entry is not None:
                        entry_url = entry.get('url', f"https://www.youtube.com/watch?v={entry['id']}")
                        urls.append(entry_url)
                        total_duration += entry.get('duration', 0)

                return urls, info.get('title', 'Collection'), total_duration
            else:
                video_url = f"https://www.youtube.com/watch?v={info['id']}"
                duration = info.get('duration', 0)
                return [video_url], info.get('title', 'Video'), duration

    except Exception as e:
        raise Exception(f"Error extracting video info: {e}")


def extract_video_metadata(url: str) -> Dict[str, Any]:
    """
    Extract basic video metadata

    Returns:
        Dictionary with video duration and title
    """
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            return {
                'duration': info.get('duration', 0),
                'title': info.get('title', 'Unknown'),
            }
    except Exception as e:
        print(f"Error extracting metadata for {url}: {e}")
        return {
            'duration': 0,
            'title': 'Unknown',
        }


def create_video_entry(video_id: int, url: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a video entry for download queue

    Args:
        video_id: Unique ID for this video
        url: YouTube video URL
        metadata: Video metadata (duration and title)

    Returns:
        Dictionary with video information
    """
    duration = metadata['duration']
    title = metadata['title']

    if duration == 0:
        raise Exception(f"Could not determine video duration for {url}")

    video_entry = {
        'video_id': video_id,
        'video_url': url,
        'video_title': title,
        'duration': duration,
        'downloaded': False,
        'transcribed': False
    }

    return video_entry


def analyze_videos(url: str) -> Dict[str, Any]:
    """
    Analyze YouTube URL and create video entries for download queue

    Args:
        url: YouTube URL (video, playlist, or channel)

    Returns:
        Dictionary with analysis results and video list
    """
    start_time = time.time()
    print(f"Analyzing: {url}", flush=True)

    video_urls, collection_title, total_duration = extract_video_info(url)

    print(f"Stage 1: Analyzing {len(video_urls)} videos", flush=True)

    all_videos = []

    for idx, video_url in enumerate(video_urls):
        try:
            metadata = extract_video_metadata(video_url)
            if metadata['duration'] == 0:
                print(f"Failed to get metadata for: {metadata['title']}", flush=True)
                continue
            video_entry = create_video_entry(idx, video_url, metadata)
            all_videos.append(video_entry)
        except Exception as e:
            print(f"Error processing {video_url}: {e}", flush=True)
            continue

    elapsed_time = time.time() - start_time
    result = {
        'source_url': url,
        'collection_title': collection_title,
        'total_videos': len(all_videos),
        'total_duration': total_duration,
        'videos': all_videos
    }

    print(f"\nStage 1 complete: {len(all_videos)} videos ({elapsed_time / 60:.1f} min)", flush=True)

    return result
"""
Transcript Database Client
Handles communication with the transcript database service
"""
import requests
import os
from typing import Optional, Dict, Any

# Database URL - can be configured via environment variable
DATABASE_URL = os.environ.get('TRANSCRIPT_DB_URL', 'http://localhost:5001')


def check_database_health() -> bool:
    """Check if database service is reachable"""
    try:
        response = requests.get(f"{DATABASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def get_transcript_from_db(video_url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch transcript from database if it exists

    Args:
        video_url: YouTube video URL

    Returns:
        Transcript data if found, None otherwise
    """
    try:
        response = requests.get(
            f"{DATABASE_URL}/transcript",
            params={'video_url': video_url},
            timeout=10
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('found'):
                return {
                    'segments': data['transcript_data'],
                    'video_title': data.get('video_title', 'Unknown'),
                    'from_database': True
                }
        return None
    except Exception as e:
        print(f"Error fetching from database: {e}")
        return None


def save_transcript_to_db(video_url: str, video_title: str, transcript_data: list) -> bool:
    """
    Save transcript to database

    Args:
        video_url: YouTube video URL
        video_title: Video title
        transcript_data: Transcript segments

    Returns:
        True if saved successfully, False otherwise
    """
    try:
        response = requests.post(
            f"{DATABASE_URL}/transcript",
            json={
                'video_url': video_url,
                'video_title': video_title,
                'transcript_data': transcript_data
            },
            timeout=30
        )

        return response.status_code == 200
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False


def get_database_stats() -> Optional[Dict[str, Any]]:
    """Get database statistics"""
    try:
        response = requests.get(f"{DATABASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

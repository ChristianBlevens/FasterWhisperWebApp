"""
Stage 3: Audio Transcription
Transcribes all downloaded audio segments
"""
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional, Callable
from faster_whisper import WhisperModel


def transcribe_audio_file(
    audio_path: str,
    whisper_model: WhisperModel,
    language: str = "en"
) -> List[Dict]:
    """
    Transcribe a single audio file

    Args:
        audio_path: Path to audio file
        whisper_model: Loaded Whisper model
        language: Language code

    Returns:
        List of segments with word-level timestamps
    """
    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )

        # Convert to serializable format
        result = []
        for segment in segments:
            seg_data = {
                'start': segment.start,
                'end': segment.end,
                'text': segment.text.strip(),
                'words': []
            }

            if hasattr(segment, 'words') and segment.words:
                for word in segment.words:
                    word_data = {
                        'word': word.word.strip(),
                        'start': word.start,
                        'end': word.end,
                        'probability': getattr(word, 'probability', 1.0)
                    }
                    seg_data['words'].append(word_data)

            result.append(seg_data)

        return result

    except Exception as e:
        print(f"Transcription error: {e}")
        return []


def transcribe_all_segments(
    videos: List[Dict],
    audio_dir: Path,
    transcripts_dir: Path,
    whisper_model: WhisperModel,
    language: str = "en",
    progress_callback: Optional[Callable] = None
) -> Dict[str, any]:
    """
    Transcribe all downloaded audio files

    Args:
        videos: List of video definitions
        audio_dir: Directory containing audio files
        transcripts_dir: Directory to save transcripts
        whisper_model: Loaded Whisper model
        language: Language code
        progress_callback: Optional callback(current, total)

    Returns:
        Dictionary with transcription results
    """
    start_time = time.time()
    transcripts_dir.mkdir(parents=True, exist_ok=True)

    transcribed_count = 0
    skipped_count = 0
    failed_count = 0

    downloaded_videos = [v for v in videos if v.get('downloaded', False)]
    pending_transcription = [v for v in downloaded_videos if not v.get('transcribed', False)]

    print(f"Stage 3: Transcribing {len(pending_transcription)} videos", flush=True)

    for idx, video in enumerate(pending_transcription):
        video_id = video['video_id']
        audio_file = video.get('audio_file')

        if not audio_file:
            print(f"Missing audio file for video {video_id}", flush=True)
            failed_count += 1
            continue

        audio_path = audio_dir / audio_file
        transcript_file = f"transcript_{video_id}.json"
        transcript_path = transcripts_dir / transcript_file

        if transcript_path.exists():
            skipped_count += 1
            video['transcribed'] = True
            video['transcript_file'] = transcript_file
            continue

        if not audio_path.exists():
            print(f"Audio file not found: {audio_path}", flush=True)
            failed_count += 1
            continue

        print(f"[{idx + 1}/{len(pending_transcription)}] Transcribing {video.get('video_title', audio_file)}", flush=True)

        video_start_time = time.time()

        transcript_data = transcribe_audio_file(
            str(audio_path),
            whisper_model,
            language
        )

        if transcript_data:
            with open(transcript_path, 'w') as f:
                json.dump({
                    'video_id': video_id,
                    'audio_file': audio_file,
                    'segments': transcript_data
                }, f, indent=2)

            transcribed_count += 1
            video['transcribed'] = True
            video['transcript_file'] = transcript_file
            video_elapsed = time.time() - video_start_time
            print(f"Transcribed: {len(transcript_data)} segments in {video_elapsed / 60:.1f} min", flush=True)
        else:
            failed_count += 1
            print(f"Transcription failed", flush=True)

        if progress_callback:
            progress_callback(idx + 1, len(pending_transcription))

    elapsed_time = time.time() - start_time
    result = {
        'transcribed': transcribed_count,
        'skipped': skipped_count,
        'failed': failed_count,
        'pending_count': len([v for v in videos if v.get('downloaded', False) and not v.get('transcribed', False)])
    }

    print(f"\nStage 3 complete: {transcribed_count} transcribed, {skipped_count} skipped, {failed_count} failed ({elapsed_time / 60:.1f} min)", flush=True)

    return result
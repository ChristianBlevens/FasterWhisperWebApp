"""
Stage 4: Transcript Parsing & Clip Planning
Parses transcripts for target words and creates clip download plan
"""
import json
import time
from pathlib import Path
from typing import List, Dict, Optional


def parse_transcript_for_words(
    transcript_data: List[Dict],
    target_words: List[str],
    padding_before: float = 0.3,
    padding_after: float = 0.3,
    merge_gap: float = 2.0
) -> List[Dict]:
    """
    Parse transcript for target words and create clip definitions

    Args:
        transcript_data: Transcript segments with word timestamps
        target_words: List of target words/phrases to find
        padding_before: Seconds to add before each word
        padding_after: Seconds to add after each word
        merge_gap: Maximum gap to merge nearby clips

    Returns:
        List of clip definitions
    """
    found_clips = []
    target_words_lower = [w.lower().strip() for w in target_words]

    for segment in transcript_data:
        if 'words' in segment and segment['words']:
            for word_obj in segment['words']:
                word_text = word_obj['word'].strip().lower()
                word_clean = ''.join(c for c in word_text if c.isalnum() or c.isspace())

                # Check for matches
                for target in target_words_lower:
                    if (word_clean == target or
                        (len(word_clean) >= 5 and len(target) >= 5 and
                         (word_clean in target or target in word_clean)) or
                        (len(word_clean) >= 6 and len(target) >= 6 and
                         abs(len(word_clean) - len(target)) <= 2 and
                         word_clean[:4] == target[:4])):

                        clip_data = {
                            'word': word_obj['word'],
                            'start': word_obj['start'],
                            'end': word_obj['end'],
                            'confidence': word_obj.get('probability', 1.0)
                        }
                        found_clips.append(clip_data)
                        break

    # Merge nearby clips
    if not found_clips:
        return []

    merged_clips = []
    current_clip = found_clips[0].copy()

    for next_clip in found_clips[1:]:
        gap = next_clip['start'] - current_clip['end']

        if merge_gap > 0 and gap <= merge_gap and gap >= -0.1:
            # Merge
            current_clip['end'] = next_clip['end']
            current_clip['word'] += f" ... {next_clip['word']}"
        else:
            # Save current and start new
            merged_clips.append(current_clip)
            current_clip = next_clip.copy()

    # Add last clip
    merged_clips.append(current_clip)

    # Add padding to all clips
    for clip in merged_clips:
        clip['start_with_padding'] = max(0, clip['start'] - padding_before)
        clip['end_with_padding'] = clip['end'] + padding_after
        clip['duration'] = clip['end_with_padding'] - clip['start_with_padding']

    return merged_clips


def create_clip_plan(
    videos: List[Dict],
    transcripts_dir: Path,
    target_words: List[str],
    padding_before: float = 0.3,
    padding_after: float = 0.3,
    merge_gap: float = 2.0
) -> Dict[str, any]:
    """
    Parse all transcripts and create comprehensive clip download plan

    Args:
        videos: List of video definitions
        transcripts_dir: Directory containing transcript files
        target_words: List of target words to find
        padding_before: Seconds before each word
        padding_after: Seconds after each word
        merge_gap: Maximum gap to merge clips

    Returns:
        Dictionary with clip plan and statistics
    """
    start_time = time.time()
    print(f"Stage 4: Planning clips for {len(target_words)} target words", flush=True)

    all_clips = []
    clip_counter = 0
    transcribed_videos = [v for v in videos if v.get('transcribed', False)]

    print(f"Processing {len(transcribed_videos)} transcripts", flush=True)

    for video in transcribed_videos:
        transcript_file = video.get('transcript_file')
        if not transcript_file:
            continue

        transcript_path = transcripts_dir / transcript_file
        if not transcript_path.exists():
            continue

        with open(transcript_path, 'r') as f:
            transcript_data = json.load(f)

        clips = parse_transcript_for_words(
            transcript_data['segments'],
            target_words,
            padding_before,
            padding_after,
            merge_gap
        )

        for clip in clips:
            clip['clip_id'] = clip_counter
            clip['video_id'] = video['video_id']
            clip['video_url'] = video['video_url']
            clip['video_title'] = video['video_title']

            # Full video downloads start at 0, so absolute timestamps = relative timestamps
            clip['absolute_start'] = clip['start_with_padding']
            clip['absolute_end'] = clip['end_with_padding']

            clip['downloaded'] = False

            all_clips.append(clip)
            clip_counter += 1

        if clips:
            print(f"Found {len(clips)} clips in video {video['video_id']}", flush=True)

    total_duration = sum(clip['duration'] for clip in all_clips)
    word_counts = {}
    for clip in all_clips:
        word = clip['word'].lower().strip()
        word_counts[word] = word_counts.get(word, 0) + 1

    elapsed_time = time.time() - start_time
    result = {
        'target_words': target_words,
        'total_clips': len(all_clips),
        'total_duration': total_duration,
        'word_counts': word_counts,
        'clips': all_clips,
        'settings': {
            'padding_before': padding_before,
            'padding_after': padding_after,
            'merge_gap': merge_gap
        }
    }

    print(f"\nStage 4 complete: {len(all_clips)} clips, {total_duration:.1f}s total ({elapsed_time / 60:.1f} min)", flush=True)

    return result
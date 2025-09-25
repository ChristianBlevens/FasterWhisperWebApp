import os
import json
import time
import threading
import pickle
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from flask import Flask, render_template, request, jsonify, send_from_directory
from faster_whisper import WhisperModel
import yt_dlp
import torch
import re
import psutil


# Import compilation functions
from compilation import (
    calculate_segments,
    get_video_duration,
    download_audio_segment,
    download_video_clip,
    parse_transcript_for_targets,
    create_compilation_video,
    get_audio_duration
)

app = Flask(__name__)

# Global variables
compilation_status = {}
whisper_model = None
COMPILATION_STATUS_FILE = 'compilation_status.pkl'
model_loaded = False
current_model_size = "medium"  # Track current loaded model

def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        print(f'✅ Using GPU: {gpu_name}')
        return 'cuda'
    else:
        print('⚠️ GPU not detected! Using CPU (will be slower)')
        return 'cpu'

def optimize_cuda_memory():
    """Configure CUDA memory allocation for maximum efficiency"""
    if torch.cuda.is_available():
        # Enable memory caching
        torch.cuda.empty_cache()

        # Set memory fraction (use 90% of available VRAM)
        torch.cuda.set_per_process_memory_fraction(0.9)

        # Enable cuDNN benchmarking for optimal algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True

        device_props = torch.cuda.get_device_properties(0)
        print(f"🚀 CUDA Memory: {device_props.total_memory / 1024**3:.1f}GB")
        print(f"🚀 CUDA Cores: {device_props.multi_processor_count}")

def load_whisper_model(model_size="medium", force_reload=False):
    """Load the optimized Whisper model with GPU acceleration"""
    global whisper_model, model_loaded, current_model_size

    # Check if same model is already loaded
    if model_loaded and whisper_model is not None and current_model_size == model_size and not force_reload:
        print(f"✅ {model_size} model already loaded")
        return True

    device = check_gpu()
    if device != 'cuda':
        print("❌ GPU required but not available. Exiting.")
        return False

    try:
        # Clear previous model if exists
        if whisper_model is not None:
            print(f"🗑️ Clearing previous {current_model_size} model from GPU memory")
            del whisper_model
            whisper_model = None
            model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Optimize CUDA memory first
        optimize_cuda_memory()

        # OPTIMIZED CONFIGURATION for maximum GPU utilization
        print(f"📥 Loading optimized Whisper model: {model_size}")
        whisper_model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="int8_float16",      # Memory optimized quantization
            device_index=0                    # Explicit GPU selection
        )
        model_loaded = True
        current_model_size = model_size
        print(f"✅ GPU-optimized {model_size} model loaded with INT8 quantization")

        # Display memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            memory_reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"🎯 GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        return True

    except Exception as e:
        print(f"❌ Failed to load {model_size} model: {e}")
        model_loaded = False
        return False

def extract_video_urls(url, max_videos=None, max_duration=None):
    """Extract video URLs from YouTube URL (single video, playlist, or channel)"""
    ydl_opts = {
        'quiet': True,
        'no_warnings': True,
        'extract_flat': True,
        'playlist_items': f'1-{max_videos}' if max_videos else None,
        # Use yt-dlp's smart extraction features
        'lazy_playlist': True,  # Don't extract all entries at once for large playlists
        'playlistreverse': False,  # Keep original order
        'playlistrandom': False,   # Don't randomize
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)

            if 'entries' in info:
                # Playlist or channel
                entries = info['entries']
                if max_videos:
                    entries = entries[:max_videos]

                urls = []
                total_duration = 0

                for entry in entries:
                    if entry is not None:
                        entry_url = entry.get('url', f"https://www.youtube.com/watch?v={entry['id']}")
                        urls.append(entry_url)

                        # Check duration limit
                        if max_duration:
                            duration = entry.get('duration', 0)
                            total_duration += duration / 60  # convert to minutes
                            if total_duration >= max_duration:
                                break

                return urls, info.get('title', 'Collection')
            else:
                # Single video
                video_url = f"https://www.youtube.com/watch?v={info['id']}"
                return [video_url], info.get('title', 'Video')
    except Exception as e:
        print(f"Error extracting URLs: {e}")
        return [], 'Unknown'

def download_youtube_audio(url, output_dir="downloads"):
    """Download audio from YouTube video"""
    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'format': 'bestaudio[ext=m4a]/bestaudio[ext=webm]/bestaudio/best',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'quiet': True,
        'no_warnings': True,
        # Use yt-dlp's advanced postprocessors for best audio quality
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '0',  # Best quality
        }],
        # Audio-specific optimizations
        'format_sort': ['abr', 'asr'],  # Prefer higher bitrate and sample rate
        'writesubtitles': False,
        'writeautomaticsub': False,
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            title = info.get('title', 'Unknown')

            # Find the downloaded file
            safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)
            audio_file = os.path.join(output_dir, f"{safe_title}.wav")

            if os.path.exists(audio_file):
                return audio_file, title

            # Sometimes the file has a different extension
            for ext in ['.wav', '.m4a', '.mp3', '.opus']:
                test_file = os.path.join(output_dir, f"{safe_title}{ext}")
                if os.path.exists(test_file):
                    return test_file, title

            return None, title

    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None, 'Unknown'

def transcribe_audio_with_progress(audio_path, language="en", task_id=None, segment_num=None, total_segments=None):
    """Transcribe audio with GPU optimization and progress tracking"""
    global whisper_model

    def progress_callback(progress):
        """Progress callback for transcription"""
        if task_id and task_id in compilation_status:
            # Calculate overall progress including segment number
            if total_segments and total_segments > 0:
                segment_progress = ((segment_num - 1) / total_segments * 100) if segment_num else 0
                current_segment_progress = progress / total_segments
                overall_progress = segment_progress + current_segment_progress
                compilation_status[task_id]['transcription_progress'] = overall_progress
                compilation_status[task_id]['stage'] = 'transcribing'
            else:
                compilation_status[task_id]['transcription_progress'] = progress
                compilation_status[task_id]['stage'] = 'transcribing'

            # Check for early termination
            if compilation_status[task_id].get('early_termination', False):
                print(f"Early termination during transcription at {progress:.1f}%")
                return True  # Signal to stop
        return False

    try:
        segments, info = whisper_model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,  # Essential for compilation
            vad_filter=True,       # Voice activity detection
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=400
            )
        )

        # Convert generator to list with progress tracking
        segments_list = []
        total_duration = info.duration if hasattr(info, 'duration') else 1

        for i, segment in enumerate(segments):
            if progress_callback((segment.end / total_duration) * 100):
                break  # Early termination requested

            segments_list.append(segment)

        return segments_list, info
    except Exception as e:
        print(f"Transcription error: {e}")
        return [], None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/create_compilation', methods=['POST'])
def create_compilation():
    """Start compilation creation process"""
    # Record the exact time the request was received
    request_received_time = time.time()

    data = request.json
    url = data.get('url')
    target_words = data.get('target_words', [])

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    if not target_words:
        return jsonify({'error': 'No target words provided'}), 400

    # Model MUST be preloaded at startup
    global whisper_model, model_loaded
    if not model_loaded or not whisper_model:
        return jsonify({'error': 'GPU-optimized model not loaded - server error'}), 500

    print(f"🎬 Starting compilation for: {url}")
    print(f"⏱️ Request received at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(request_received_time))}")

    try:
        # Extract video URLs
        urls, collection_title = extract_video_urls(url)
        if not urls:
            return jsonify({'error': 'Could not extract video URLs'}), 400

        # Generate task ID
        task_id = f"compilation_{int(time.time())}"

        # Start compilation task in background
        options = {
            'target_words': target_words,
            'padding_before': data.get('padding_before', 1.0),
            'padding_after': data.get('padding_after', 1.0),
            'merge_gap': data.get('merge_gap', 2.0),
            'max_length': data.get('max_length')
        }

        thread = threading.Thread(
            target=process_compilation_task,
            args=(task_id, urls, options, request_received_time)
        )
        thread.daemon = True
        thread.start()

        return jsonify({
            'task_id': task_id,
            'video_title': collection_title,
            'total_videos': len(urls)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/compilation_status/<task_id>')
def get_compilation_status(task_id):
    """Get compilation processing status"""
    if task_id not in compilation_status:
        return jsonify({'error': 'Compilation task not found'}), 404
    return jsonify(compilation_status[task_id])

@app.route('/stop_compilation/<task_id>', methods=['POST'])
def stop_compilation(task_id):
    """Request early termination of compilation task"""
    if task_id not in compilation_status:
        return jsonify({'error': 'Compilation task not found'}), 404

    compilation_status[task_id]['early_termination'] = True
    save_compilation_status()
    return jsonify({'message': 'Stop requested'})

@app.route('/compilation_video/<path:filename>')
def serve_compilation_video(filename):
    """Serve compilation video for preview"""
    return send_from_directory('compilations', filename)

@app.route('/download_compilation/<path:filename>')
def download_compilation(filename):
    """Download compilation video file"""
    return send_from_directory('compilations', filename, as_attachment=True)

@app.route('/test_transcript', methods=['POST'])
def test_transcript():
    """Simple transcript test for single videos"""
    # Record the exact time the request was received
    request_start_time = time.time()

    data = request.json
    url = data.get('url')

    if not url:
        return jsonify({'error': 'No URL provided'}), 400

    # Check if it's a single video (not playlist/channel)
    try:
        urls, title = extract_video_urls(url)
        if len(urls) != 1:
            return jsonify({'error': 'Test transcript only works with single videos'}), 400
    except:
        return jsonify({'error': 'Invalid YouTube URL'}), 400

    # Model MUST be preloaded at startup
    global whisper_model, model_loaded
    if not model_loaded or not whisper_model:
        return jsonify({'error': 'GPU-optimized model not loaded - server error'}), 500

    try:
        # Download audio
        print(f"📥 Downloading audio for transcript test: {url}")
        audio_file, title = download_youtube_audio(url)

        if not audio_file:
            return jsonify({'error': 'Failed to download audio'}), 500

        # Transcribe
        print(f"🎙️ Transcribing: {title}")
        transcribe_start = time.time()
        segments, info = transcribe_audio_with_progress(audio_file, language="en")
        transcribe_duration = time.time() - transcribe_start

        # Clean up audio file
        if os.path.exists(audio_file):
            os.remove(audio_file)

        if not segments:
            return jsonify({'error': 'Failed to transcribe audio'}), 500

        # Create simple transcript with segment-level timestamps
        transcript_text = ""
        for segment in segments:
            transcript_text += f"[{segment.start:.1f}s - {segment.end:.1f}s] {segment.text.strip()}\n"

        # Create word-level timestamps section
        word_timestamps = "\n" + "=" * 50 + "\n"
        word_timestamps += "WORD-LEVEL TIMESTAMPS\n"
        word_timestamps += "=" * 50 + "\n\n"

        for segment in segments:
            if hasattr(segment, 'words') and segment.words:
                word_timestamps += f"Segment [{segment.start:.1f}s - {segment.end:.1f}s]:\n"
                for word_obj in segment.words:
                    word_clean = word_obj.word.strip()
                    confidence = getattr(word_obj, 'probability', 1.0)
                    word_timestamps += f"  [{word_obj.start:.2f}s - {word_obj.end:.2f}s] '{word_clean}' (confidence: {confidence:.2f})\n"
                word_timestamps += "\n"

        # Save transcript file
        os.makedirs('transcripts', exist_ok=True)
        safe_title = re.sub(r'[<>:"/\\|?*]', '_', title)[:50]
        transcript_filename = f"{safe_title}_transcript.txt"
        transcript_path = f"transcripts/{transcript_filename}"

        with open(transcript_path, 'w', encoding='utf-8') as f:
            f.write(f"Transcript for: {title}\n")
            f.write("=" * 50 + "\n\n")
            f.write("SENTENCE-LEVEL TRANSCRIPT\n")
            f.write("=" * 30 + "\n\n")
            f.write(transcript_text)
            f.write(word_timestamps)

        # Calculate overall timing
        total_request_time = time.time() - request_start_time

        print(f"⏱️ Transcription completed in {transcribe_duration:.1f}s")
        print(f"⏱️ OVERALL REQUEST TIME: {total_request_time:.1f}s")

        return jsonify({
            'success': True,
            'title': title,
            'transcript_file': transcript_filename,
            'duration': info.duration if hasattr(info, 'duration') else 0,
            'transcription_time': transcribe_duration,
            'total_request_time': total_request_time
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_transcript/<path:filename>')
def download_transcript(filename):
    """Download transcript file"""
    return send_from_directory('transcripts', filename, as_attachment=True)

@app.route('/switch_model', methods=['POST'])
def switch_model():
    """Switch to a different Whisper model"""
    global current_model_size

    data = request.json
    new_model = data.get('model_size')

    if not new_model:
        return jsonify({'error': 'No model specified'}), 400

    valid_models = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
    if new_model not in valid_models:
        return jsonify({'error': f'Invalid model. Must be one of: {valid_models}'}), 400

    try:
        print(f"🔄 Switching from {current_model_size} to {new_model} model...")
        success = load_whisper_model(new_model, force_reload=True)

        if success:
            return jsonify({
                'success': True,
                'model': new_model,
                'message': f'{new_model} model loaded successfully'
            })
        else:
            return jsonify({
                'success': False,
                'error': f'Failed to load {new_model} model',
                'current_model': current_model_size
            }), 500

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'current_model': current_model_size
        }), 500

def process_compilation_task(task_id, urls, options, request_received_time=None):
    """Segmented processing task for compilation video creation"""
    global whisper_model, compilation_status

    # Track processing start time (when thread actually begins processing)
    processing_start_time = time.time()

    # Initialize status
    compilation_status[task_id] = {
        'status': 'processing',
        'stage': 'initializing',
        'current_video': 0,
        'total_videos': len(urls),
        'current_video_title': '',
        'current_segment': 0,
        'total_segments': 0,
        'segments_info': None,
        'transcription_progress': 0.0,
        'words_found': [],
        'clips_extracted': [],
        'compilation_length': 0.0,
        'word_count': 0,
        'early_termination': False,
        'compilation_path': None,
        'video_file': None,
        'final_duration': 0,
        'errors': [],
        'start_time': processing_start_time,
        'request_received_time': request_received_time if request_received_time else processing_start_time,
        'processing_start_time': processing_start_time,
        'elapsed_time': 0,
        'total_transcription_time': 0
    }

    # Extract parameters
    target_words = options.get('target_words', [])
    max_length = options.get('max_length')
    padding_before = options.get('padding_before', 1.0)
    padding_after = options.get('padding_after', 1.0)
    merge_gap = options.get('merge_gap', 2.0)

    all_clips = []

    for video_idx, url in enumerate(urls):
        # Check early termination
        if compilation_status[task_id]['early_termination']:
            print(f"Early termination requested at video {video_idx+1}/{len(urls)}")
            break

        # Check length limit
        if max_length and compilation_status[task_id]['compilation_length'] >= max_length:
            print(f"Max compilation length reached: {max_length}s")
            break

        compilation_status[task_id]['current_video'] = video_idx + 1

        try:
            # Get video info for segmentation
            compilation_status[task_id]['stage'] = 'analyzing'
            duration, title = get_video_duration(url)
            if duration == 0:
                print(f"⚠️ Could not get video duration for {url}")
                compilation_status[task_id]['errors'].append(f"Failed to analyze: {url}")
                continue

            # Calculate segments
            segments = calculate_segments(duration)
            compilation_status[task_id]['total_segments'] = len(segments)
            compilation_status[task_id]['current_video_title'] = title
            save_compilation_status()

            print(f"🎬 Processing {title} in {len(segments)} segments")

            # Process each segment
            for seg_idx, (start_time, end_time) in enumerate(segments):
                # Check early termination
                if compilation_status[task_id]['early_termination']:
                    print(f"Early termination during segment {seg_idx+1}/{len(segments)}")
                    break

                compilation_status[task_id]['current_segment'] = seg_idx + 1
                save_compilation_status()

                print(f"📥 Downloading audio segment {seg_idx+1}/{len(segments)} ({start_time}-{end_time}s)")

                # Download audio segment
                audio_file = download_audio_segment(
                    url, start_time, end_time,
                    f"downloads/temp_segment_{task_id}_{seg_idx}.wav"
                )

                if not audio_file:
                    compilation_status[task_id]['errors'].append(
                        f"Failed to download audio segment {seg_idx+1} from {title}"
                    )
                    continue

                # Transcribe segment with timing
                transcribe_start = time.time()
                transcript_segments, info = transcribe_audio_with_progress(
                    audio_file,
                    language="en",
                    task_id=task_id,
                    segment_num=seg_idx + 1,
                    total_segments=len(segments)
                )
                transcribe_duration = time.time() - transcribe_start
                compilation_status[task_id]['total_transcription_time'] += transcribe_duration
                compilation_status[task_id]['elapsed_time'] = time.time() - compilation_status[task_id]['request_received_time']
                print(f"⏱️ Transcribed segment {seg_idx+1} in {transcribe_duration:.1f}s")
                save_compilation_status()

                if not transcript_segments:
                    compilation_status[task_id]['errors'].append(
                        f"Failed to transcribe segment {seg_idx+1} from {title}"
                    )
                    if os.path.exists(audio_file):
                        os.remove(audio_file)
                    continue

                # Parse transcript for target words immediately
                found_clips = parse_transcript_for_targets(
                    transcript_segments, target_words, merge_gap
                )

                # Download video clips for found words (in parallel)
                if found_clips:
                    compilation_status[task_id]['stage'] = 'downloading'
                    compilation_status[task_id]['current_clip'] = 0
                    compilation_status[task_id]['total_clips'] = len(found_clips)
                    compilation_status[task_id]['elapsed_time'] = time.time() - compilation_status[task_id]['request_received_time']
                    save_compilation_status()

                    # Prepare clip jobs for parallel downloading
                    clip_jobs = []
                    for clip_idx, clip in enumerate(found_clips):
                        # Check limits
                        if max_length and compilation_status[task_id]['compilation_length'] >= max_length:
                            break

                        # Adjust timestamps for segment offset
                        actual_start = start_time + clip['start']
                        actual_end = start_time + clip['end']
                        clip_duration = (actual_end - actual_start) + padding_before + padding_after

                        # Create clips directory
                        os.makedirs('clips', exist_ok=True)
                        clip_path = f"clips/{task_id}_{video_idx}_{seg_idx}_{clip_idx}.mp4"

                        clip_jobs.append({
                            'index': len(clip_jobs),  # Add chronological index
                            'url': url,
                            'start': actual_start,
                            'end': actual_end,
                            'path': clip_path,
                            'word': clip['word'],
                            'duration': clip_duration,
                            'title': title,
                            'padding_before': padding_before,
                            'padding_after': padding_after
                        })

                    print(f"🚀 Downloading {len(clip_jobs)} clips in parallel...")

                    # Download clips in parallel (max 4 concurrent downloads)
                    def download_clip_job(job):
                        return download_video_clip(
                            job['url'], job['start'], job['end'], job['path'],
                            job['padding_before'], job['padding_after']
                        ), job

                    successful_clips = []
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        # Submit all jobs
                        future_to_job = {executor.submit(download_clip_job, job): job for job in clip_jobs}

                        # Process completed downloads
                        for future in as_completed(future_to_job):
                            job = future_to_job[future]
                            try:
                                success, job_data = future.result()
                                compilation_status[task_id]['current_clip'] += 1

                                if success:
                                    clip_info = {
                                        'index': job['index'],  # Add chronological index
                                        'path': job_data['path'],
                                        'word': job_data['word'],
                                        'duration': job_data['duration'],
                                        'source_video': job_data['title'],
                                        'timestamp': job_data['start']
                                    }

                                    successful_clips.append(clip_info)
                                    print(f"✅ Downloaded clip {compilation_status[task_id]['current_clip']}/{len(clip_jobs)}: {job_data['word']} ({job_data['start']:.1f}-{job_data['end']:.1f}s)")

                                    # Update status
                                    compilation_status[task_id]['clips_extracted'].append(clip_info)
                                    compilation_status[task_id]['compilation_length'] += job_data['duration']
                                    compilation_status[task_id]['word_count'] += 1
                                    compilation_status[task_id]['words_found'].append(job_data['word'])
                                else:
                                    print(f"❌ Failed to download clip: {job_data['word']} ({job_data['start']:.1f}-{job_data['end']:.1f}s)")

                                save_compilation_status()

                            except Exception as e:
                                print(f"❌ Error processing clip: {e}")

                    # Sort successful clips by chronological index and add to main list
                    successful_clips.sort(key=lambda x: x['index'])
                    all_clips.extend(successful_clips)

                    # Update final status
                    compilation_status[task_id]['segments_info'] = {
                        'found_clips': len(all_clips),
                        'total_duration': compilation_status[task_id]['compilation_length']
                    }
                    save_compilation_status()

                # Clean up audio segment immediately
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    print(f"🗑️ Cleaned up audio segment {seg_idx+1}")

        except Exception as e:
            error_msg = f"Error processing {url}: {str(e)}"
            print(f"❌ {error_msg}")
            compilation_status[task_id]['errors'].append(error_msg)
            save_compilation_status()

    # Create final compilation video
    if all_clips:
        compilation_status[task_id]['stage'] = 'compiling'
        compilation_status[task_id]['elapsed_time'] = time.time() - compilation_status[task_id]['request_received_time']
        save_compilation_status()

        compilation_start = time.time()
        print(f"🎬 Creating compilation from {len(all_clips)} clips...")

        os.makedirs('compilations', exist_ok=True)
        output_filename = f"compilation_{task_id}.mp4"
        output_path = f"compilations/{output_filename}"

        final_video, total_duration = create_compilation_video(all_clips, output_path, max_length)

        # Cleanup clip files
        for clip in all_clips:
            if os.path.exists(clip['path']):
                os.remove(clip['path'])

        if final_video:
            compilation_status[task_id]['compilation_path'] = final_video
            compilation_status[task_id]['video_file'] = output_filename
            compilation_status[task_id]['status'] = 'completed'
            compilation_status[task_id]['final_duration'] = total_duration

            # Calculate final overall timing
            final_time = time.time()
            compilation_time = final_time - compilation_start
            total_request_time = final_time - compilation_status[task_id]['request_received_time']
            processing_time = final_time - compilation_status[task_id]['processing_start_time']

            compilation_status[task_id]['total_request_time'] = total_request_time
            compilation_status[task_id]['processing_time'] = processing_time
            compilation_status[task_id]['compilation_time'] = compilation_time

            print(f"✅ Compilation created: {final_video} ({total_duration:.1f}s)")
            print(f"⏱️ OVERALL REQUEST TIME: {total_request_time:.1f}s")
            print(f"   ├─ Processing time: {processing_time:.1f}s")
            print(f"   ├─ Transcription time: {compilation_status[task_id]['total_transcription_time']:.1f}s")
            print(f"   └─ Compilation time: {compilation_time:.1f}s")
        else:
            compilation_status[task_id]['status'] = 'error'
            compilation_status[task_id]['errors'].append("Failed to create compilation video")
    else:
        compilation_status[task_id]['status'] = 'completed'
        compilation_status[task_id]['errors'].append("No target words found in any video")
        print("⚠️ No target words found in any video")

    compilation_status[task_id]['end_time'] = time.time()

    # Calculate final timing if not already done
    if 'total_request_time' not in compilation_status[task_id]:
        total_request_time = compilation_status[task_id]['end_time'] - compilation_status[task_id]['request_received_time']
        processing_time = compilation_status[task_id]['end_time'] - compilation_status[task_id]['processing_start_time']
        compilation_status[task_id]['total_request_time'] = total_request_time
        compilation_status[task_id]['processing_time'] = processing_time
        print(f"⏱️ Final request-to-completion time: {total_request_time:.1f}s")

    save_compilation_status()

def save_compilation_status():
    """Save compilation status to disk"""
    global compilation_status
    try:
        with open(COMPILATION_STATUS_FILE, 'wb') as f:
            pickle.dump(compilation_status, f)
    except Exception as e:
        print(f"Error saving compilation status: {e}")

def load_compilation_status():
    """Load compilation status from disk"""
    global compilation_status
    try:
        if os.path.exists(COMPILATION_STATUS_FILE):
            with open(COMPILATION_STATUS_FILE, 'rb') as f:
                compilation_status = pickle.load(f)
            print(f"Loaded {len(compilation_status)} compilation tasks from disk")
    except Exception as e:
        print(f"Error loading compilation status: {e}")
        compilation_status = {}

def monitor_gpu_usage():
    """Monitor GPU usage in background"""
    def gpu_monitor():
        while True:
            try:
                if torch.cuda.is_available():
                    memory_allocated = torch.cuda.memory_allocated() / 1024**3
                    memory_reserved = torch.cuda.memory_reserved() / 1024**3

                    # Get GPU utilization using nvidia-smi
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True
                    )

                    if result.returncode == 0:
                        gpu_util, mem_used, mem_total = result.stdout.strip().split(', ')
                        print(f"🔥 GPU: {gpu_util}% | VRAM: {mem_used}MB/{mem_total}MB | PyTorch: {memory_allocated:.1f}GB allocated")
                time.sleep(30)  # Monitor every 30 seconds
            except Exception as e:
                print(f"GPU monitoring error: {e}")
                time.sleep(30)

    monitor_thread = threading.Thread(target=gpu_monitor, daemon=True)
    monitor_thread.start()

# Initialize application on module import (works with both direct run and Gunicorn)
print("🚀 Starting GPU-Accelerated Video Compilation Server...")

# Load compilation status on startup
load_compilation_status()

# Start GPU monitoring
monitor_gpu_usage()

# Preload the Whisper model only once
load_whisper_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
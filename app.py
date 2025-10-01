"""
Multi-Stage Video Compilation Flask Application
Integrates all 6 stages with project management
"""
import os
import torch
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from faster_whisper import WhisperModel

# Import project model
from models.project import Project

# Import stage modules
from stages.stage1_video_analysis import analyze_videos
from stages.stage2_audio_download import download_videos_batch
from stages.stage3_transcription import transcribe_all_segments
from stages.stage4_clip_planning import create_clip_plan
from stages.stage5_clip_download import download_clips_batch
from stages.stage6_compilation import compile_video

# Import database client
from transcript_database_client import get_transcript_from_db, save_transcript_to_db, check_database_health

app = Flask(__name__)

# Global variables
whisper_model = None
model_loaded = False
current_model_size = "medium"


def check_gpu():
    """Check GPU availability"""
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        gpu_name = torch.cuda.get_device_name(0)
        print(f'Using GPU: {gpu_name}')
        return 'cuda'
    else:
        print('WARNING: GPU not detected! Using CPU (will be slower)')
        return 'cpu'


def optimize_cuda_memory():
    """Configure CUDA memory allocation for maximum efficiency"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.set_per_process_memory_fraction(0.9)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        device_props = torch.cuda.get_device_properties(0)
        print(f"CUDA Memory: {device_props.total_memory / 1024**3:.1f}GB")
        print(f"CUDA Cores: {device_props.multi_processor_count}")


def load_whisper_model(model_size="medium", force_reload=False):
    """Load the optimized Whisper model with GPU acceleration"""
    global whisper_model, model_loaded, current_model_size

    if model_loaded and whisper_model is not None and current_model_size == model_size and not force_reload:
        print(f"{model_size} model already loaded")
        return True

    device = check_gpu()
    if device != 'cuda':
        print("ERROR: GPU required but not available.")
        return False

    try:
        if whisper_model is not None:
            print(f"Clearing previous {current_model_size} model from GPU memory")
            del whisper_model
            whisper_model = None
            model_loaded = False
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        optimize_cuda_memory()

        print(f"Loading optimized Whisper model: {model_size}")
        whisper_model = WhisperModel(
            model_size,
            device="cuda",
            compute_type="int8_float16",
            device_index=0
        )
        model_loaded = True
        current_model_size = model_size
        print(f"GPU-optimized {model_size} model loaded with INT8 quantization")

        #if torch.cuda.is_available():
        #    memory_allocated = torch.cuda.memory_allocated() / 1024**3
        #    memory_reserved = torch.cuda.memory_reserved() / 1024**3
        #    print(f"GPU Memory - Allocated: {memory_allocated:.1f}GB, Reserved: {memory_reserved:.1f}GB")

        return True

    except Exception as e:
        print(f"ERROR: Failed to load {model_size} model: {e}")
        model_loaded = False
        return False


# Routes

@app.route('/')
def index():
    return render_template('index.html')


# Project Management Routes

@app.route('/api/projects', methods=['GET'])
def list_projects():
    """List all projects"""
    projects = Project.list_projects()
    return jsonify({'projects': projects})


@app.route('/api/projects/create', methods=['POST'])
def create_project():
    """Create new project"""
    project = Project.create_new()
    return jsonify({
        'project_id': project.project_id,
        'metadata': project.metadata
    })


@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project_status(project_id):
    """Get project status"""
    try:
        project = Project(project_id)
        return jsonify({
            'project_id': project_id,
            'metadata': project.metadata,
            'videos': project.load_videos(),
            'clip_plan': project.load_clip_plan()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/projects/<project_id>/rename', methods=['POST'])
def rename_project(project_id):
    """Rename project"""
    try:
        import shutil
        from pathlib import Path

        data = request.get_json()
        new_title = data.get('new_title', '').strip()

        if not new_title:
            return jsonify({'error': 'New title required'}), 400

        base_dir = Path('projects')
        old_path = base_dir / project_id
        new_path = base_dir / new_title

        if not old_path.exists():
            return jsonify({'error': 'Project not found'}), 404

        if new_path.exists():
            return jsonify({'error': 'Project with that name already exists'}), 400

        shutil.move(str(old_path), str(new_path))

        project = Project(new_title)
        project.metadata['project_id'] = new_title
        project._save_metadata(project.metadata)

        return jsonify({
            'project_id': new_title,
            'metadata': project.metadata
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Stage 1: Segmentation

@app.route('/api/projects/<project_id>/stage1/analyze', methods=['POST'])
def stage1_analyze(project_id):
    """Stage 1: Analyze videos and check database for existing transcripts"""
    try:
        project = Project(project_id)
        data = request.json

        url = data.get('url')
        use_database = data.get('use_database', True)

        if not url:
            return jsonify({'error': 'URL required'}), 400

        # Clear Stage 1 output (stage 1 always clears)
        project.clear_stage_output(1)

        project.update_stage_status('segmentation', 'processing', 0)

        result = analyze_videos(url)
        videos = result['videos']

        transcripts_dir = project.get_transcripts_dir()
        transcripts_dir.mkdir(parents=True, exist_ok=True)

        db_fetched_count = 0

        # Check database for existing transcripts if enabled
        if use_database:
            print("Stage 1: Checking database connection...", flush=True)
            db_healthy = check_database_health()
            if db_healthy:
                print("Stage 1: Database connected successfully", flush=True)
            else:
                print("Stage 1: Database not available - continuing without it", flush=True)

        if use_database and check_database_health():
            print("Stage 1: Querying database for existing transcripts", flush=True)
            import json

            for video in videos:
                video_url = video.get('video_url')
                db_transcript = get_transcript_from_db(video_url)

                if db_transcript:
                    print(f"Found transcript in database for video {video['video_id']}: {video.get('video_title', 'Unknown')}", flush=True)

                    # Save transcript to Stage 3 output
                    transcript_file = f"video_{video['video_id']}_transcript.json"
                    transcript_path = transcripts_dir / transcript_file

                    with open(transcript_path, 'w') as f:
                        json.dump({
                            'segments': db_transcript['segments'],
                            'language': 'en'
                        }, f, indent=2)

                    # Mark as already transcribed (skip Stage 2 and 3)
                    video['transcribed'] = True
                    video['transcript_file'] = transcript_file
                    video['downloaded'] = True  # Mark as downloaded so Stage 2 skips it
                    db_fetched_count += 1

        project.metadata['source_url'] = result['source_url']
        project.metadata['collection_title'] = result['collection_title']
        project.save_videos(videos)

        result['db_fetched_count'] = db_fetched_count

        project.update_stage_status('segmentation', 'completed', 100,
                                     total_videos=result['total_videos'],
                                     total_duration=result['total_duration'],
                                     db_fetched_count=db_fetched_count)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        project.update_stage_status('segmentation', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Stage 2: Audio Download



@app.route('/api/projects/<project_id>/stage2/download', methods=['POST'])
def stage2_download(project_id):
    """Stage 2: Download full video audio with anti-throttling"""
    try:
        project = Project(project_id)
        data = request.json

        gb_limit = data.get('gb_limit', 5.0)

        videos = project.load_videos()
        if not videos:
            return jsonify({'error': 'No videos found. Run Stage 1 first.'}), 400

        project.update_stage_status('audio_download', 'processing', 0)

        def progress_callback(current, total, downloaded_gb):
            progress = (current / total) * 100
            project.update_stage_status('audio_download', 'processing', progress,
                                        downloaded_gb=downloaded_gb)

        result = download_videos_batch(
            videos,
            project.get_audio_segments_dir(),
            gb_limit,
            progress_callback
        )

        project.save_videos(videos)

        status = 'completed' if result['pending_count'] == 0 else 'partial'
        project.update_stage_status('audio_download', status, 100, **result)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        project.update_stage_status('audio_download', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Stage 3: Transcription

@app.route('/api/projects/<project_id>/stage3/transcribe', methods=['POST'])
def stage3_transcribe(project_id):
    """Stage 3: Transcribe audio segments"""
    global whisper_model, model_loaded

    if not model_loaded or not whisper_model:
        return jsonify({'error': 'Whisper model not loaded'}), 500

    try:
        project = Project(project_id)
        data = request.json

        language = data.get('language', 'en')
        use_database = data.get('use_database', True)

        videos = project.load_videos()
        if not videos:
            return jsonify({'error': 'No videos found. Please run Stage 1 first.'}), 400

        project.update_stage_status('transcription', 'processing', 0)

        transcripts_dir = project.get_transcripts_dir()
        audio_dir = project.get_audio_segments_dir()

        db_fetched_count = 0
        db_saved_count = 0

        # Try to fetch from database if enabled
        if use_database:
            print("Stage 3: Checking database connection...", flush=True)
            db_healthy = check_database_health()
            if db_healthy:
                print("Stage 3: Database connected successfully", flush=True)
            else:
                print("Stage 3: Database not available - continuing without it", flush=True)

        if use_database and check_database_health():
            print("Stage 3: Checking for existing transcripts in database", flush=True)
            for video in videos:
                if video.get('transcribed'):
                    continue

                video_url = video.get('video_url')
                db_transcript = get_transcript_from_db(video_url)

                if db_transcript:
                    print(f"Found transcript in database for video {video['video_id']}", flush=True)

                    # Save to local file
                    transcript_file = f"video_{video['video_id']}_transcript.json"
                    transcript_path = transcripts_dir / transcript_file

                    import json
                    with open(transcript_path, 'w') as f:
                        json.dump({
                            'segments': db_transcript['segments'],
                            'language': language
                        }, f, indent=2)

                    video['transcribed'] = True
                    video['transcript_file'] = transcript_file
                    db_fetched_count += 1

            if db_fetched_count > 0:
                project.save_videos(videos)
                print(f"Fetched {db_fetched_count} transcripts from database", flush=True)

        def progress_callback(current, total):
            progress = (current / total) * 100
            project.update_stage_status('transcription', 'processing', progress)

        result = transcribe_all_segments(
            videos,
            audio_dir,
            transcripts_dir,
            whisper_model,
            language,
            progress_callback
        )

        # Save new transcripts to database if enabled
        if use_database and check_database_health():
            print("Saving new transcripts to database", flush=True)
            for video in videos:
                if not video.get('transcribed') or not video.get('transcript_file'):
                    continue

                transcript_path = transcripts_dir / video['transcript_file']
                if transcript_path.exists():
                    import json
                    with open(transcript_path, 'r') as f:
                        transcript_data = json.load(f)

                    video_title = video.get('video_title', 'Unknown')
                    video_url = video['video_url']

                    print(f"Sending transcript to database: {video_title} ({video_url})", flush=True)

                    if save_transcript_to_db(
                        video_url,
                        video_title,
                        transcript_data.get('segments', [])
                    ):
                        db_saved_count += 1
                        print(f"Successfully saved transcript to database: {video_title}", flush=True)
                    else:
                        print(f"Failed to save transcript to database: {video_title}", flush=True)

            if db_saved_count > 0:
                print(f"Total saved to database: {db_saved_count} transcripts", flush=True)

        project.save_videos(videos)

        status = 'completed' if result['pending_count'] == 0 else 'partial'
        result['db_fetched_count'] = db_fetched_count
        result['db_saved_count'] = db_saved_count
        project.update_stage_status('transcription', status, 100, **result)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        project.update_stage_status('transcription', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Stage 4: Clip Planning

@app.route('/api/projects/<project_id>/stage4/plan', methods=['POST'])
def stage4_plan(project_id):
    """Stage 4: Parse transcripts and plan clips"""
    try:
        project = Project(project_id)
        data = request.json

        target_words = data.get('target_words', [])
        padding_before = data.get('padding_before', 0.3)
        padding_after = data.get('padding_after', 0.3)
        merge_gap = data.get('merge_gap', 2.0)

        if not target_words:
            return jsonify({'error': 'Target words required'}), 400

        # Clear Stage 4 output (stage 4 always clears)
        project.clear_stage_output(4)

        videos = project.load_videos()
        if not videos:
            return jsonify({'error': 'No videos found. Please run Stage 1 first.'}), 400

        project.update_stage_status('clip_planning', 'processing', 0)

        result = create_clip_plan(
            videos,
            project.get_transcripts_dir(),
            target_words,
            padding_before,
            padding_after,
            merge_gap
        )

        project.save_clip_plan(result['clips'])

        project.update_stage_status('clip_planning', 'completed', 100,
                                    total_clips=result['total_clips'],
                                    total_duration=result['total_duration'],
                                    word_counts=result['word_counts'])

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        project.update_stage_status('clip_planning', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Stage 5: Clip Download

@app.route('/api/projects/<project_id>/stage5/download', methods=['POST'])
def stage5_download(project_id):
    """Stage 5: Download video clips"""
    try:
        project = Project(project_id)
        data = request.json

        gb_limit = data.get('gb_limit', 10.0)
        max_workers = data.get('max_workers', 32)

        clips = project.load_clip_plan()
        if not clips:
            return jsonify({'error': 'No clip plan found. Run Stage 4 first.'}), 400

        project.update_stage_status('clip_download', 'processing', 0)

        def progress_callback(current, total, downloaded_gb):
            progress = (current / total) * 100
            project.update_stage_status('clip_download', 'processing', progress,
                                        downloaded_gb=downloaded_gb)

        result = download_clips_batch(
            clips,
            project.get_clips_dir(),
            gb_limit,
            max_workers,
            progress_callback
        )

        project.save_clip_plan(clips)

        status = 'completed' if result['pending_count'] == 0 else 'partial'
        project.update_stage_status('clip_download', status, 100, **result)

        return jsonify({
            'success': True,
            'result': result
        })

    except Exception as e:
        project.update_stage_status('clip_download', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Stage 6: Compilation

@app.route('/api/projects/<project_id>/stage6/compile', methods=['POST'])
def stage6_compile(project_id):
    """Stage 6: Compile final video"""
    try:
        project = Project(project_id)
        data = request.json

        max_length = data.get('max_length')

        clips = project.load_clip_plan()
        if not clips:
            return jsonify({'error': 'No clips found'}), 400

        project.update_stage_status('compilation', 'processing', 0)

        result = compile_video(
            clips,
            project.get_clips_dir(),
            project.get_compilation_path(),
            max_length
        )

        if result['success']:
            project.update_stage_status('compilation', 'completed', 100,
                                        output_path=result['output_path'],
                                        final_duration=result['final_duration'],
                                        file_size_gb=result['file_size_gb'])
        else:
            project.update_stage_status('compilation', 'error', 0,
                                        error=result.get('error', 'Unknown error'))

        return jsonify({
            'success': result['success'],
            'result': result
        })

    except Exception as e:
        project.update_stage_status('compilation', 'error', 0, error=str(e))
        return jsonify({'error': str(e)}), 500


# Download final compilation

@app.route('/api/projects/<project_id>/download', methods=['GET'])
def download_final_compilation(project_id):
    """Download final compilation video"""
    try:
        project = Project(project_id)
        compilation_path = project.get_compilation_path()

        if not compilation_path.exists():
            return jsonify({'error': 'Compilation not found'}), 404

        return send_from_directory(
            compilation_path.parent,
            compilation_path.name,
            as_attachment=True
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/api/projects/<project_id>/download/<filename>', methods=['GET'])
def download_alt_compilation(project_id, filename):
    """Download alt pipeline compilation video"""
    try:
        project = Project(project_id)

        # Only allow downloading of alt compilation files
        if filename not in ['final_compilation_alt.mp4']:
            return jsonify({'error': 'File not allowed'}), 403

        file_path = project.get_alt_stage_output_dir('compilation') / filename

        if not file_path.exists():
            return jsonify({'error': 'File not found'}), 404

        return send_from_directory(
            str(file_path.parent),
            file_path.name,
            as_attachment=True
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Clear stage output endpoints

@app.route('/api/projects/<project_id>/stage<int:stage_number>/clear', methods=['POST'])
def clear_stage_output(project_id, stage_number):
    """Clear all files from a stage's output directory"""
    try:
        if stage_number < 1 or stage_number > 6:
            return jsonify({'error': 'Invalid stage number'}), 400

        project = Project(project_id)

        result = project.clear_stage_output(stage_number)

        return jsonify({
            'success': True,
            'deleted_files': result['deleted_files'],
            'freed_gb': result['freed_gb']
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/projects/<project_id>/stage<int:stage_number>/info', methods=['GET'])
def get_stage_info(project_id, stage_number):
    """Get information about a stage's output directory"""
    try:
        if stage_number < 1 or stage_number > 6:
            return jsonify({'error': 'Invalid stage number'}), 400

        project = Project(project_id)

        return jsonify({
            'stage_number': stage_number,
            'file_count': project.get_stage_file_count(stage_number),
            'size_gb': project.get_stage_size(stage_number)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Model management

@app.route('/api/switch_model', methods=['POST'])
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


# Pipeline logging endpoint
@app.route('/api/log-pipeline', methods=['POST'])
def log_pipeline():
    """Log pipeline completion with timing"""
    data = request.json
    elapsed_seconds = data.get('elapsed_seconds', 0)
    stages_run = data.get('stages_run', [])

    print(f"\nPipeline complete: {len(stages_run)} stages ({elapsed_seconds / 60:.1f} min)", flush=True)

    return jsonify({'success': True})


# Initialize application
print("=" * 60, flush=True)
print("FasterWhisperWebApp v2.1.0 - 2025-09-30", flush=True)
print("=" * 60, flush=True)

# Preload the Whisper model
load_whisper_model()

if __name__ == '__main__':
    print("\n" + "=" * 60, flush=True)
    print("FRONTEND READY - Application available at http://localhost:5000", flush=True)
    print("=" * 60 + "\n", flush=True)
    app.run(host='0.0.0.0', port=5000, debug=False)
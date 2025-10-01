# Multi-Stage Video Compilation System

## Overview

This is a complete refactor of the Faster-Whisper video compilation application into a modular, multi-stage pipeline with independent stage control and organized output folders.

## New Directory Structure

```
FasterWhisperWebApp/
├── app.py                          # New multi-stage Flask application
├── app_original.py                 # Original monolithic version (backup)
├── models/
│   ├── __init__.py
│   └── project.py                  # Project management with organized outputs
├── stages/
│   ├── __init__.py
│   ├── stage1_segmentation.py      # Video analysis & segmentation
│   ├── stage2_audio_download.py    # Audio downloading with GB limits
│   ├── stage3_transcription.py     # Audio transcription
│   ├── stage4_clip_planning.py     # Clip planning from transcripts
│   ├── stage5_clip_download.py     # Video clip downloading
│   └── stage6_compilation.py       # Final video compilation
├── templates/
│   ├── index.html                  # New multi-tab single-page interface
│   └── index_original.html         # Original interface (backup)
├── static/
│   └── js/
│       └── app.js                  # Frontend JavaScript for all stages
└── projects/                       # Project data (auto-created)
    └── {project_id}/
        ├── project.json            # Project metadata & status
        ├── stage1_segments/        # Segmentation: segments.json
        ├── stage2_audio/           # Audio: *.wav audio files
        ├── stage3_transcripts/     # Transcripts: *.json files
        ├── stage4_clip_plan/       # Clip plan: clip_plan.json
        ├── stage5_clips/           # Clips: *.mp4 video clips
        └── stage6_compilation/     # Compilation: final_compilation.mp4
```

## Key Features

### 1. Stage-Specific Output Folders
- **Separate folder per stage**: `stage1_output/`, `stage2_output/`, etc.
- **Clear organization**: Each stage writes to its own dedicated folder
- **Independent cleanup**: Clear any stage without affecting others

### 2. Independent Stage Control
- Each stage can be run separately
- Resume capability for stages with GB limits (2 & 5)
- Clear output buttons for each stage
- Real-time file count and size tracking

### 3. Project Management
- Create multiple projects
- Switch between projects
- Persistent project state
- Status tracking per stage

### 4. Stage Info Bars
Each tab displays:
- **Output Files**: Number of files in that stage's output
- **Output Size**: Total size in GB
- **Clear Output Button**: Delete all files with confirmation

## Stage Pipeline

```
Stage 1: Video Analysis & Segmentation
   ↓ stage1_segments/segments.json

Stage 2: Audio Segment Downloading (GB limit, resumable)
   ↓ stage2_audio/*.wav

Stage 3: Audio Transcription
   ↓ stage3_transcripts/*.json

Stage 4: Transcript Parsing & Clip Planning
   ↓ stage4_clip_plan/clip_plan.json

Stage 5: Video Clip Downloading (GB limit, resumable)
   ↓ stage5_clips/*.mp4

Stage 6: Final Video Compilation
   ↓ stage6_compilation/final_compilation.mp4
```

## Running the Application

### Option 1: Direct Python
```bash
cd /workspace/FasterWhisperWebApp
python3 app.py
```

### Option 2: Docker
```bash
docker-compose up --build
```

Access at: **http://localhost:5000**

## Usage Workflow

1. **Create/Select Project**
   - Click "New Project" or select existing from dropdown

2. **Stage 1: Segmentation**
   - Enter YouTube URL
   - Set segment duration (default 600s = 10min)
   - Click "Analyze & Segment"
   - Output: `stage1_segments/segments.json`

3. **Stage 2: Audio Download**
   - Set GB limit (default 5GB)
   - Click "Download Audio"
   - Can run multiple times until all downloaded
   - Output: `stage2_audio/*.wav`

4. **Stage 3: Transcription**
   - Select language
   - Click "Transcribe Audio"
   - Output: `stage3_transcripts/*.json`

5. **Stage 4: Clip Planning**
   - Enter target words (comma-separated)
   - Set padding before/after
   - Set merge gap
   - Click "Plan Clips"
   - Output: `stage4_clip_plan/clip_plan.json`

6. **Stage 5: Clip Download**
   - Set GB limit (default 10GB)
   - Set parallel workers (1-8)
   - Click "Download Clips"
   - Can run multiple times until all downloaded
   - Output: `stage5_clips/*.mp4`

7. **Stage 6: Compilation**
   - Optionally set max length
   - Click "Create Compilation"
   - Output: `stage6_compilation/final_compilation.mp4`
   - Click download button to get final video

## API Endpoints

### Project Management
- `GET /api/projects` - List all projects
- `POST /api/projects/create` - Create new project
- `GET /api/projects/<id>` - Get project status

### Stage Execution
- `POST /api/projects/<id>/stage1/analyze` - Run segmentation
- `POST /api/projects/<id>/stage2/download` - Download audio
- `POST /api/projects/<id>/stage3/transcribe` - Transcribe audio
- `POST /api/projects/<id>/stage4/plan` - Plan clips
- `POST /api/projects/<id>/stage5/download` - Download clips
- `POST /api/projects/<id>/stage6/compile` - Create compilation

### Stage Management
- `GET /api/projects/<id>/stage<N>/info` - Get stage file count & size
- `POST /api/projects/<id>/stage<N>/clear` - Clear stage output

### Download
- `GET /api/projects/<id>/download` - Download final compilation

## Status Badges

Each tab shows a colored badge indicating stage status:
- **Gray**: Pending (not started)
- **Orange**: Processing (in progress)
- **Yellow**: Partial (some items completed, more to do)
- **Green**: Completed (all done)
- **Red**: Error

## Clear Output Feature

Each stage has a "🗑️ Clear Output" button that:
1. Shows confirmation dialog
2. Deletes all files in that stage's output folder
3. Resets stage status to "pending"
4. Shows count of deleted files and freed space
5. Updates the info bar display

## Files Removed

- `compilation.py` - Old monolithic module (no longer used)
  - Functions integrated into respective stage modules

## Advantages Over Original

1. **Modularity**: Each stage is independent
2. **Resumability**: Stages 2 & 5 can pause and resume with GB limits
3. **Organization**: Stage-specific output folders
4. **Visibility**: Real-time tracking of file count and sizes
5. **Control**: Clear outputs individually without affecting other stages
6. **Flexibility**: Run stages in any order, re-run as needed
7. **Scalability**: Easy to add new stages or modify existing ones

## Requirements

Same as original:
- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Runtime
- Or Python 3.8+ with dependencies from `requirements.txt`

## Testing

```bash
# Start the application
python3 app.py

# Open browser
http://localhost:5000

# Create a test project
1. Click "New Project"
2. Use a short YouTube video URL
3. Run each stage sequentially
4. Check outputs/ folder structure
5. Test clear output buttons
6. Test GB limits on stages 2 & 5
```

## Migration from Original

The new system is completely independent. To switch:
1. Original app backed up as `app_original.py`
2. Original template backed up as `index_original.html`
3. New app is now main `app.py`
4. Projects created with old system are separate
5. No data migration needed
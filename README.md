# FasterWhisperWebApp

A web application that automatically finds and extracts specific words or phrases from YouTube videos, channels, or playlists, then compiles them into a single video.

## Quick Start

### What You Need
1. **NVIDIA GPU**
2. **Docker Desktop** - [Download here](https://www.docker.com/products/docker-desktop/)
2. **This App** - Clone from GitHub (or download directly at https://github.com/ChristianBlevens/FasterWhisperWebApp):
   ```bash
   git clone https://github.com/ChristianBlevens/FasterWhisperWebApp.git
   ```

### How to Run

**GPU Mode (requires NVIDIA GPU):**
```bash
docker-compose --profile gpu up --build
```
Or simply:
```bash
docker-compose up --build
```

**CPU Mode (no GPU required, uses database transcripts only):**
```bash
docker-compose --profile cpu up --build
```

#### Startup Messages

Wait for one of these messages:

**GPU Mode:**
```
FRONTEND READY - Application available at http://localhost:5000
All 6 stages available (GPU mode)
```

**CPU Mode:**
```
FRONTEND READY - Application available at http://localhost:5000
CPU MODE: Stages 1,4,5,6 available (requires database transcripts)
Stages 2 and 3 are disabled - no transcription available
```

Then open your browser and go to: **http://localhost:5000**

### How to Use
1. **Enter a YouTube URL** (video, channel, or playlist)
2. **Enter target words** (the words or phrases you want to find)
3. **Click "Process Pipeline"**
4. Wait for the app to automatically process through all 6 stages
5. **Download your final compilation**

That's it! The app does everything automatically.

---

## Features

- **Automatic Transcription**: Uses Faster Whisper AI to transcribe videos (GPU mode only)
- **Smart Word Search**: Finds exact words or phrases with context
- **Cloud Database**: Saves transcriptions to avoid re-processing the same videos
- **Parallel Processing**: Downloads clips concurrently for faster processing
- **GPU/CPU Modes**: Works with or without NVIDIA GPU
- **Comment Section**: Share feedback at the bottom of the page
- **Customizable Parameters**: Adjust padding, merge gaps, and more

---

## GPU vs CPU Mode

### GPU Mode (Default)
- **Requirements**: NVIDIA GPU with CUDA support
- **Available Stages**: All 6 stages (1-6)
- **Transcription**: Full transcription capability with Faster Whisper
- **Use Case**: Process any YouTube video, even without existing transcripts

### CPU Mode
- **Requirements**: No GPU needed, runs on any system
- **Available Stages**: Stages 1, 4, 5, 6 only
- **Transcription**: Disabled - must use videos with existing database transcripts
- **Use Case**: Process videos that have been transcribed by other users or previously transcribed
- **Limitations**:
  - Stage 2 (Audio Download) is disabled
  - Stage 3 (Transcription) is disabled
  - Must rely on cloud database for transcripts

**How CPU Mode Works:**
1. Stage 1 checks database and downloads any existing transcripts
2. If all videos have transcripts in database, you can proceed to Stage 4
3. Stage 4, 5, 6 work normally (clip planning, download, compilation)

**When to use CPU Mode:**
- You don't have an NVIDIA GPU
- You're processing popular videos likely to have transcripts in the database
- You're re-processing videos you've already transcribed before
- You're collaborating with others who have uploaded transcripts

---

## Frontend Interface

### Pipeline Tab
- **YouTube URL Input**: Enter video, channel, or playlist URL
- **Target Words Input**: Enter words/phrases to search for (comma-separated)
- **Advanced Parameters**:
  - **Padding Before**: Seconds to include before each word (default: 0.3)
  - **Padding After**: Seconds to include after each word (default: 0.3)
  - **Merge Gap**: Maximum gap between clips before merging (default: 2.0)
- **Stage Selection Checkboxes**: Choose which stages to run in sequence
  - Check only the stages you want to execute
  - Stages run in order from 1 to 6
  - Example: Check stages 4, 5, 6 to search for new words and compile without re-downloading/transcribing
  - Example: Check only stage 6 to re-compile existing clips with different settings
- **Process Pipeline Button**: Starts the selected stages in sequential order
- **Progress Bar**: Shows current stage and progress

### Individual Stage Tabs
Each stage has its own tab where you can:
- Run just that stage independently
- View results for that stage
- Download outputs
- Adjust stage-specific parameters

**Use cases for individual tabs**:
- Re-transcribe with different Whisper model settings
- Search for different target words without re-downloading
- Re-compile clips in different order

### Project Management
- Multiple projects can exist simultaneously
- Each project has its own folder under `projects/`
- Projects can be renamed
- Stage outputs can be cleared individually to free space

---

## The Pipeline (6 Stages)

The app processes videos through 6 sequential stages. Each stage must complete before the next begins.

### Stage 1: Video Analysis & Database Check
**What it does**: Parses YouTube URL and checks database for existing transcripts
**Input**: YouTube URL (video/channel/playlist)
**Output**: Video metadata + any existing transcripts from database
**Details**:
- Extracts all video URLs from playlists or channels
- Gets metadata (title, duration) for each video
- Checks cloud database for existing transcripts
- If transcript exists, downloads it and skips to Stage 4
- Saves video list for next stages

### Stage 2: Audio Download
**What it does**: Downloads full video audio files
**Input**: Video list from Stage 1
**Output**: Audio files (.wav format) in `projects/{project_name}/audio_segments/` folder
**Details**:
- Uses yt-dlp with aria2c for fast, anti-throttled downloads
- Downloads worst quality audio (sufficient for transcription)
- Converts to 16kHz WAV format for Whisper
- Respects GB limit (default: 5 GB)
- Skips videos already downloaded

### Stage 3: Transcription
**What it does**: Converts audio to text using Faster Whisper AI
**Input**: Audio files from Stage 2
**Output**: Transcript JSON files in `projects/{project_name}/transcripts/` folder
**Details**:
- Uses Faster Whisper `medium` model by default (GPU-accelerated)
- Word-level timestamps with confidence scores
- Voice Activity Detection (VAD) to filter silence
- Checks database again before transcribing (in case another user uploaded it)
- Uploads new transcripts to cloud database for future reuse
- Skips videos that came from database

### Stage 4: Clip Planning
**What it does**: Finds target words in transcripts and plans clips
**Input**: Target words + transcripts from Stage 3
**Output**: Clip plan JSON in `projects/{project_name}/clip_plan.json`
**Details**:
- Searches for exact word matches (case-insensitive)
- Adds padding before/after each word (default: 0.3 seconds)
- Merges clips that are close together (default: 2.0 second gap)
- Each clip includes start time, end time, source video, and matched text
- Calculates word frequency statistics

### Stage 5: Clip Download
**What it does**: Downloads individual video clips from YouTube
**Input**: Clip plan from Stage 4
**Output**: Video clip files in `projects/{project_name}/clips/` folder
**Details**:
- Downloads clips directly from YouTube using yt-dlp
- Parallel downloads (default: 32 concurrent)
- Normalizes all clips to 30 FPS constant framerate
- Re-encodes to H.264/AAC for compatibility
- Respects GB limit (default: 10 GB)
- Skips clips already downloaded

### Stage 6: Compilation
**What it does**: Combines all clips into one final video
**Input**: Video clips from Stage 5
**Output**: Final compilation in `projects/{project_name}/final_compilation.mp4`
**Details**:
- Concatenates clips in chronological order using FFmpeg
- Normalizes all clips to 1920x1080 resolution (adds letterboxing if needed)
- Automatically handles mixed resolutions and aspect ratios
- Skips clips without audio streams
- Creates downloadable MP4 file

---

## Output Files and Folders

```
FasterWhisperWebApp/
└── projects/
    └── {project_name}/
        ├── metadata.json              # Project settings and status
        ├── videos.json                # Video list with download status
        ├── clip_plan.json             # Planned clips with timestamps
        ├── audio_segments/            # Stage 2 output
        │   └── video_0_Title.wav      # Downloaded audio files
        ├── transcripts/               # Stage 3 output
        │   └── transcript_0.json      # Word-level transcripts
        ├── clips/                     # Stage 5 output
        │   └── clip_00001.mp4         # Individual video clips
        └── final_compilation.mp4      # Stage 6 output
```

### File Naming Conventions
- **Audio**: `video_{id}_{title}.wav`
- **Transcripts**: `transcript_{id}.json`
- **Clips**: `clip_{id:05d}.mp4`
- **Compilation**: `final_compilation.mp4`

### Transcript JSON Format
```json
{
  "video_id": 0,
  "audio_file": "video_0_Title.wav",
  "segments": [
    {
      "start": 0.0,
      "end": 5.2,
      "text": "Hello world",
      "words": [
        {
          "word": "Hello",
          "start": 0.0,
          "end": 0.5,
          "probability": 0.98
        },
        {
          "word": "world",
          "start": 0.6,
          "end": 1.2,
          "probability": 0.95
        }
      ]
    }
  ]
}
```

---

## Configuration

### Environment Variables
Located in `docker-compose.yml`:

```yaml
TRANSCRIPT_DB_URL: http://mycomments.duckdns.org:5001
```
- **Purpose**: URL of the cloud database for storing/retrieving transcripts
- **Current Setup**: Hosted on Hetzner server, accessible publicly
- **Change if**: You want to host your own database instance

### Default Parameters
Can be changed in frontend or `stages/stage4_clip_planning.py`:

- **Padding Before/After**: 0.3 seconds (how much video to include around each word)
- **Merge Gap**: 2.0 seconds (clips closer than this are merged together)
- **Whisper Model**: `medium` (can change in Stage 3 settings)
- **Stage 2 GB Limit**: 5.0 GB
- **Stage 5 GB Limit**: 10.0 GB
- **Max Workers**: 32 parallel clip downloads

---

## Cloud Database

The app uses a cloud-hosted transcript database to avoid re-processing videos.

**How it works**:
1. **Stage 1**: Before analyzing videos, the app checks if transcripts already exist in the database
2. **Stage 3**: Before transcribing, the app checks again (in case another user uploaded it since Stage 1)
3. **Stage 3**: After transcribing new videos, transcripts are uploaded to the database
4. This saves significant time and computational resources

**Database Location**: `http://mycomments.duckdns.org:5001`

**Benefits**:
- Faster processing for previously transcribed videos
- Share transcripts across multiple users
- Persistent storage even if local files are deleted
- Skips entire transcription stage when transcripts exist

**Database Service**: Runs in a separate Docker container on a Hetzner server with SQLite backend.

**Privacy Note**: Video URLs and transcripts are stored on the public database. Do not use this app with private or sensitive videos.

---

## Comment Section

At the bottom of the page, there's an embedded comment section powered by a custom CommentSectionWebApp.

**Features**:
- Leave feedback or suggestions
- No account required
- Shared across all users

**How it works**:
- Loads from: `https://cdn.jsdelivr.net/gh/ChristianBlevens/CommentSectionWebApp@main/embed.js`
- Connects to: `https://mycomments.duckdns.org`
- Page ID: `FasterWhisperWebApp`

---

## Technical Details

### Architecture
- **Backend**: Python Flask application
- **Transcription**: Faster Whisper (OpenAI Whisper optimized with CTranslate2)
- **Video Processing**: FFmpeg
- **Video Download**: yt-dlp with aria2c
- **Database**: SQLite with Flask REST API
- **Frontend**: Vanilla JavaScript + HTML/CSS
- **Containerization**: Docker + Docker Compose

### Docker Services
The app runs two Docker containers:

1. **Main App** (`docker-compose.yml`):
   - Flask web server on port 5000
   - Loads Faster Whisper model on startup
   - Processes all 6 pipeline stages
   - Mounts local `projects/` folder for persistent storage

2. **Database Service** (separate server):
   - Flask REST API on port 5001
   - SQLite database for transcript storage
   - Hosted at `http://mycomments.duckdns.org:5001`

### FFmpeg Processing

**Stage 5 (Clip Normalization)**:
```bash
ffmpeg -i input.mp4 -vf "fps=30,scale=trunc(iw/2)*2:trunc(ih/2)*2" -c:v libx264 -preset fast -crf 23 -bf 0 -c:a aac output.mp4
```
- Forces 30 FPS constant framerate
- No B-frames to prevent PTS/DTS reordering issues
- Ensures all clips have compatible framerates

**Stage 6 (Compilation)**:
```bash
ffmpeg -i clip1.mp4 -i clip2.mp4 ... -filter_complex "[0:v]scale=1920:1080...[v0];[1:v]scale=1920:1080...[v1];...;[v0][0:a][v1][1:a]...concat=n={N}:v=1:a=1[outv][outa]" -map "[outv]" -map "[outa]" output.mp4
```
- Scales all clips to 1920x1080 with letterboxing
- Concatenates video and audio streams
- Ensures uniform resolution and audio streams

### API Endpoints

**Database Service**:
- `GET /api/transcript/{video_id}` - Retrieve transcript by video URL hash
- `POST /api/transcript` - Upload transcript
- `GET /health` - Health check

**Main App**:
- `GET /` - Frontend interface
- `POST /api/projects/create` - Create new project
- `GET /api/projects` - List all projects
- `POST /api/projects/{id}/stage1/analyze` - Run Stage 1
- `POST /api/projects/{id}/stage2/download` - Run Stage 2
- `POST /api/projects/{id}/stage3/transcribe` - Run Stage 3
- `POST /api/projects/{id}/stage4/plan` - Run Stage 4
- `POST /api/projects/{id}/stage5/download` - Run Stage 5
- `POST /api/projects/{id}/stage6/compile` - Run Stage 6
- `GET /api/projects/{id}/download` - Download final compilation
- `POST /api/projects/{id}/stage{N}/clear` - Clear stage output

### Whisper Model Options

Available models (sorted by speed):
- `tiny` - Fastest, least accurate
- `base` - Fast, low accuracy
- `small` - Balanced
- `medium` - **Default** - Good balance of speed and accuracy
- `large` - Slow, very accurate
- `large-v2` - Slower, more accurate
- `large-v3` - Slowest, most accurate

Change model in Stage 3 settings or via API:
```bash
POST /api/switch_model
{"model_size": "large-v3"}
```

---

## Troubleshooting

### "Database not available"
- Database service may be down
- Check if `http://mycomments.duckdns.org:5001/health` responds
- App will still work, just slower (no transcript caching)

### "FFmpeg error: Stream specifier ':a' matches no streams"
- One of the clips has no audio
- App automatically skips these clips now
- Check `clips/` folder for source video

### "Resolution mismatch" error
- Videos from different sources have different resolutions
- App automatically normalizes to 1920x1080 now
- All clips will have letterboxing/pillarboxing if needed

### Frontend not loading
- Wait for "FRONTEND READY" message in terminal
- Whisper model loads on startup in GPU mode (can take 30-60 seconds)
- Don't access `localhost:5000` until this message appears

### Docker build fails
- Ensure Docker Desktop is running
- Check available disk space (Whisper model is ~3GB for GPU mode)
- Try `docker-compose down` then `docker-compose up --build`

### GPU not detected
- If you want GPU mode, ensure NVIDIA GPU with CUDA support is installed
- Install NVIDIA Docker runtime
- Check GPU is accessible: `nvidia-smi`
- If no GPU, use CPU mode: `docker-compose --profile cpu up --build`

### CPU Mode: "Stage 2/3 disabled" error
- You're running in CPU mode which doesn't support transcription
- Check if video has transcript in database by running Stage 1 first
- If no transcript exists, you'll need to switch to GPU mode
- Restart with: `docker-compose --profile gpu up --build`

### Which mode am I running?
- Check the startup logs for "Mode: GPU" or "Mode: CPU"
- GPU mode shows "All 6 stages available"
- CPU mode shows "Stages 1,4,5,6 available"

### How to switch between modes
- To use GPU mode: `docker-compose --profile gpu up --build` or `docker-compose up --build`
- To use CPU mode: `docker-compose --profile cpu up --build`

### Slow downloads
- YouTube may be throttling
- App uses aria2c with 16 connections to bypass throttling
- Try running at different times of day

---

## Development

### Project Structure
```
FasterWhisperWebApp/
├── app.py                      # Main Flask application
├── docker-compose.yml          # Docker configuration
├── Dockerfile                  # Docker build instructions
├── requirements.txt            # Python dependencies
├── models/
│   └── project.py             # Project management class
├── stages/                     # Pipeline stage implementations
│   ├── stage1_video_analysis.py
│   ├── stage2_audio_download.py
│   ├── stage3_transcription.py
│   ├── stage4_clip_planning.py
│   ├── stage5_clip_download.py
│   └── stage6_compilation.py
├── templates/
│   └── index.html             # Frontend UI
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── app.js
├── transcript_database_client.py  # Database API client
└── projects/                   # User projects (mounted volume)
```

### Adding Features
1. **New Pipeline Stage**: Create `stages/stageN_name.py` with appropriate function
2. **Frontend UI**: Add section to `templates/index.html` and handlers in `static/js/app.js`
3. **API Endpoint**: Add route in `app.py`
4. **Project Structure**: Modify `models/project.py` if new folders needed

### Running Without Docker
Not recommended, but possible:
1. Install Python 3.9+
2. Install FFmpeg, aria2c, and yt-dlp
3. Install NVIDIA CUDA toolkit
4. `pip install -r requirements.txt`
5. `python app.py`

---

## Performance Tips

- **Use database caching**: Don't disable database checks - they save significant time
- **Adjust GB limits**: Stage 2 and Stage 5 have GB limits to prevent filling disk
- **Parallel downloads**: Stage 5 uses 32 parallel downloads by default - increase if you have high bandwidth
- **Whisper model**: Use `medium` for best balance, `small` for speed, `large-v3` for accuracy
- **GPU required**: App requires NVIDIA GPU - CPU transcription is extremely slow

---

## Credits

- **Whisper**: OpenAI's speech recognition model
- **Faster Whisper**: CTranslate2 optimization by Guillaume Klein
- **yt-dlp**: YouTube download tool
- **aria2c**: Multi-connection download utility
- **FFmpeg**: Video processing library
- **Comment Section**: Custom CommentSectionWebApp

---

## License

This project is open source and available under the MIT License.

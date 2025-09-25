# Faster-Whisper AI Video Tools

A GPU-accelerated Docker container that creates compilation videos by finding target words in YouTube videos using OpenAI's Faster-Whisper.

## Features

- **GPU-Accelerated**: Uses NVIDIA CUDA for fast transcription
- **YouTube Support**: Videos, playlists, and channels
- **Word-Level Precision**: Finds exact moments when target words are spoken
- **Video Compilation**: Creates compilation videos with automatic clipping and merging
- **Smart Segmentation**: Handles long videos efficiently with 10-minute segments
- **Real-time Progress**: Live status updates with early termination support
- **Web Interface**: Clean Flask frontend on port 5000

## Prerequisites

- NVIDIA GPU with CUDA support
- Docker with NVIDIA Container Runtime
- Docker Compose

### Install NVIDIA Container Runtime

```bash
# Ubuntu/Debian
curl -s -L https://nvidia.github.io/nvidia-container-runtime/gpgkey | sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-container-runtime/$distribution/nvidia-container-runtime.list | sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get install nvidia-container-runtime
sudo systemctl restart docker
```

## Quick Start

1. **Clone/Download** this repository

2. **Build and run**:
   ```bash
   docker-compose up --build
   ```

3. **Access the web interface**:
   Open http://localhost:5000 in your browser

## Usage

### Web Interface

1. Enter a YouTube URL:
   - Single video: `https://www.youtube.com/watch?v=VIDEO_ID`
   - Playlist: `https://www.youtube.com/playlist?list=PLAYLIST_ID`
   - Channel: `https://www.youtube.com/@channelname` or `https://www.youtube.com/c/channelname`

2. Enter target words:
   - Comma-separated list: `hello, world, amazing, cool`
   - System finds exact moments when these words are spoken

3. Configure settings:
   - **Padding**: Extra seconds before/after each word
   - **Merge Gap**: Clips within this gap are merged together
   - **Max Length**: Optional compilation time limit

4. Click "Create Compilation" and monitor progress

### Output

You'll get a single MP4 compilation video containing all moments where your target words were spoken, with seamless transitions and preserved audio.

## Model Sizes

| Model | Speed | Accuracy | VRAM Usage |
|-------|-------|----------|------------|
| tiny  | Fastest | Lowest | ~1GB |
| base  | Fast | Low | ~1GB |
| small | Medium | Medium | ~2GB |
| medium | Balanced | Good | ~5GB |
| large | Slow | High | ~10GB |
| large-v3 | Slowest | Best | ~10GB |

## API Endpoints

- `POST /create_compilation` - Start compilation creation
- `GET /compilation_status/<task_id>` - Check compilation progress
- `POST /stop_compilation/<task_id>` - Stop processing early
- `GET /compilation_video/<filename>` - Preview video
- `GET /download_compilation/<filename>` - Download compilation

## Configuration

Environment variables in `docker-compose.yml`:
- `NVIDIA_VISIBLE_DEVICES=all` - Use all GPUs
- `NVIDIA_DRIVER_CAPABILITIES=compute,utility` - Required capabilities

## Troubleshooting

### GPU Not Detected
```bash
# Check NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04 nvidia-smi

# If this fails, reinstall NVIDIA Container Runtime
```

### Out of Memory
- Process shorter videos or reduce max compilation length
- Check available VRAM: `nvidia-smi`
- Monitor memory usage during processing

### No Clips Found
- Try broader target words or variations
- Check that audio is clear and in the expected language
- Verify target words are actually spoken in the video

### Slow Processing
- Verify GPU usage: `nvidia-smi` while processing
- Check that CUDA is available in container logs
- Ensure sufficient VRAM for model

## Development

To modify the application:

1. Edit `app.py` for backend changes
2. Edit `templates/index.html` for frontend changes
3. Rebuild: `docker-compose up --build`

## Support

This container automatically:
- Downloads Whisper models on first use
- Manages temporary audio/video files with cleanup
- Handles video segmentation for memory efficiency
- Provides real-time progress tracking with early termination

All processing is done locally - no data leaves your machine.
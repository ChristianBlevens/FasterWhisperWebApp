"""
Project data model for multi-stage compilation
"""
import os
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any


class Project:
    """Manages project data across all compilation stages"""

    def __init__(self, project_id: str, base_dir: str = "projects"):
        self.project_id = project_id
        self.base_dir = Path(base_dir)
        self.project_dir = self.base_dir / project_id

        # Create project directory structure with descriptive stage folders
        self.project_dir.mkdir(parents=True, exist_ok=True)
        (self.project_dir / "stage1_segments").mkdir(exist_ok=True)
        (self.project_dir / "stage2_audio").mkdir(exist_ok=True)
        (self.project_dir / "stage3_transcripts").mkdir(exist_ok=True)
        (self.project_dir / "stage4_clip_plan").mkdir(exist_ok=True)
        (self.project_dir / "stage5_clips").mkdir(exist_ok=True)
        (self.project_dir / "stage6_compilation").mkdir(exist_ok=True)

        self.metadata_file = self.project_dir / "project.json"

        # Load or initialize metadata
        self.metadata = self._load_or_init_metadata()

    def _load_or_init_metadata(self) -> Dict[str, Any]:
        """Load existing metadata or create new"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            metadata = {
                'project_id': self.project_id,
                'created_at': time.time(),
                'updated_at': time.time(),
                'source_url': None,
                'source_title': None,
                'stages': {
                    'segmentation': {'status': 'pending', 'progress': 0},
                    'audio_download': {'status': 'pending', 'progress': 0},
                    'transcription': {'status': 'pending', 'progress': 0},
                    'clip_planning': {'status': 'pending', 'progress': 0},
                    'clip_download': {'status': 'pending', 'progress': 0},
                    'compilation': {'status': 'pending', 'progress': 0}
                },
                'alt_stages': {
                    'transcript_analysis': {'status': 'pending', 'progress': 0},
                    'sentence_audio': {'status': 'pending', 'progress': 0}
                },
                'pipeline_type': 'small'  # 'small' or 'large'
            }
            self._save_metadata(metadata)
            return metadata

    def _save_metadata(self, metadata: Dict[str, Any]):
        """Save metadata to disk"""
        metadata['updated_at'] = time.time()
        with open(self.metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

    def update_stage_status(self, stage: str, status: str, progress: float = None, **kwargs):
        """Update stage status and progress"""
        self.metadata['stages'][stage]['status'] = status
        if progress is not None:
            self.metadata['stages'][stage]['progress'] = progress

        # Add any additional stage-specific data
        for key, value in kwargs.items():
            self.metadata['stages'][stage][key] = value

        self._save_metadata(self.metadata)

    def get_stage_status(self, stage: str) -> Dict[str, Any]:
        """Get current status of a stage"""
        return self.metadata['stages'].get(stage, {})

    def get_stage_output_dir(self, stage_number: int) -> Path:
        """Get output directory for a specific stage"""
        stage_names = {
            1: "stage1_segments",
            2: "stage2_audio",
            3: "stage3_transcripts",
            4: "stage4_clip_plan",
            5: "stage5_clips",
            6: "stage6_compilation"
        }
        return self.project_dir / stage_names[stage_number]

    def save_videos(self, videos: List[Dict[str, Any]]):
        """Save video list to stage1_segments (renamed but keeping folder name for compatibility)"""
        output_file = self.get_stage_output_dir(1) / "videos.json"
        with open(output_file, 'w') as f:
            json.dump(videos, f, indent=2)

    def load_videos(self) -> List[Dict[str, Any]]:
        """Load video list from stage1_segments"""
        videos_file = self.get_stage_output_dir(1) / "videos.json"
        if videos_file.exists():
            with open(videos_file, 'r') as f:
                return json.load(f)
        return []

    def save_clip_plan(self, clip_plan: List[Dict[str, Any]]):
        """Save clip planning results to stage4_clip_plan"""
        output_file = self.get_stage_output_dir(4) / "clip_plan.json"
        with open(output_file, 'w') as f:
            json.dump(clip_plan, f, indent=2)

    def load_clip_plan(self) -> List[Dict[str, Any]]:
        """Load clip planning results from stage4_clip_plan"""
        clip_plan_file = self.get_stage_output_dir(4) / "clip_plan.json"
        if clip_plan_file.exists():
            with open(clip_plan_file, 'r') as f:
                return json.load(f)
        return []

    def get_audio_segments_dir(self) -> Path:
        """Get audio segments directory (stage2_audio)"""
        return self.get_stage_output_dir(2)

    def get_transcripts_dir(self) -> Path:
        """Get transcripts directory (stage3_transcripts)"""
        return self.get_stage_output_dir(3)

    def get_clips_dir(self) -> Path:
        """Get clips directory (stage5_clips)"""
        return self.get_stage_output_dir(5)

    def get_compilation_path(self) -> Path:
        """Get final compilation output path (stage6_compilation)"""
        return self.get_stage_output_dir(6) / "final_compilation.mp4"

    def get_stage_size(self, stage_number: int) -> float:
        """Get size of stage output directory in GB"""
        dir_path = self.get_stage_output_dir(stage_number)
        total_size = 0
        for entry in dir_path.rglob('*'):
            if entry.is_file():
                total_size += entry.stat().st_size
        return total_size / (1024 ** 3)  # Convert to GB

    def get_stage_file_count(self, stage_number: int) -> int:
        """Get count of files in stage output directory"""
        dir_path = self.get_stage_output_dir(stage_number)
        if not dir_path.exists():
            return 0
        return len([f for f in dir_path.rglob('*') if f.is_file()])

    def clear_stage_output(self, stage_number: int) -> Dict[str, Any]:
        """
        Clear all files from a stage's output directory

        Args:
            stage_number: Stage number (1-6)

        Returns:
            Dictionary with deletion results
        """
        dir_path = self.get_stage_output_dir(stage_number)

        if not dir_path.exists():
            return {'deleted_files': 0, 'freed_gb': 0.0}

        # Calculate size before deletion
        size_before = self.get_stage_size(stage_number)
        file_count = self.get_stage_file_count(stage_number)

        # Delete all files in directory
        for item in dir_path.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)

        # Reset stage status to pending
        stage_names = {
            1: 'segmentation',
            2: 'audio_download',
            3: 'transcription',
            4: 'clip_planning',
            5: 'clip_download',
            6: 'compilation'
        }

        if stage_number in stage_names:
            self.update_stage_status(stage_names[stage_number], 'pending', 0)

        return {
            'deleted_files': file_count,
            'freed_gb': size_before
        }

    @classmethod
    def list_projects(cls, base_dir: str = "projects") -> List[str]:
        """List all project IDs"""
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        return [d.name for d in base_path.iterdir() if d.is_dir()]

    @classmethod
    def create_new(cls, base_dir: str = "projects") -> 'Project':
        """Create a new project with unique ID"""
        project_id = f"project_{int(time.time() * 1000)}"
        return cls(project_id, base_dir)


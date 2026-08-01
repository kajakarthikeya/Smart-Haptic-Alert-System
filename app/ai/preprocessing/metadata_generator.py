"""Metadata Generator for saving preprocessed dataset manifest in JSON format."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional
import wave

from config import settings
from app.utils.logger import get_logger
from app.ai.preprocessing.models import ProcessedFileMetadata

logger = get_logger(__name__)


def compute_sha256(file_path: Path) -> str:
    """Computes SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


class MetadataGenerator:
    """Generates and exports structured JSON metadata for preprocessed audio files."""

    def __init__(self, processed_dir: Optional[Path] = None) -> None:
        """Initializes MetadataGenerator.

        Args:
            processed_dir: Output directory for processed dataset.
                           Defaults to settings.preprocessing.processed_dir.
        """
        self._processed_dir = processed_dir or settings.preprocessing.processed_dir
        logger.info(f"MetadataGenerator initialized for directory: {self._processed_dir}")

    def create_file_metadata(
        self,
        output_path: Path,
        class_label: str,
        duration_sec: float = 4.0,
        sample_rate: int = 22050,
        channels: int = 1,
    ) -> ProcessedFileMetadata:
        """Generates ProcessedFileMetadata object for a single preprocessed file.

        Args:
            output_path: Path to preprocessed output file.
            class_label: Target sound class label.
            duration_sec: Duration in seconds.
            sample_rate: Sample rate in Hz.
            channels: Channel count.

        Returns:
            ProcessedFileMetadata instance.
        """
        file_size = output_path.stat().st_size if output_path.exists() else 0
        sha256_hash = compute_sha256(output_path) if output_path.exists() else ""
        timestamp = datetime.now(timezone.utc).isoformat()

        return ProcessedFileMetadata(
            file_name=output_path.name,
            class_label=class_label,
            output_path=str(output_path.resolve()),
            duration_sec=duration_sec,
            sample_rate=sample_rate,
            channels=channels,
            file_size_bytes=file_size,
            sha256_hash=sha256_hash,
            processing_timestamp=timestamp,
        )

    def export_summary_json(
        self, metadata_records: List[ProcessedFileMetadata], json_filename: str = "preprocessed_metadata.json"
    ) -> Path:
        """Exports list of ProcessedFileMetadata records to a JSON file.

        Args:
            metadata_records: List of ProcessedFileMetadata objects.
            json_filename: File name for summary JSON.

        Returns:
            Path object of written JSON file.
        """
        self._processed_dir.mkdir(parents=True, exist_ok=True)
        json_path = self._processed_dir / json_filename

        data = {
            "total_files": len(metadata_records),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "files": [record.to_dict() for record in metadata_records],
        }

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported preprocessed metadata JSON report ({len(metadata_records)} files) to {json_path}")
        return json_path

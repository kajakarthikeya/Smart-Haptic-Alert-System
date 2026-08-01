"""Domain Data Models for Audio Preprocessing."""

from dataclasses import dataclass, field, asdict
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class RawAudioData:
    """Dataclass holding raw audio waveform data and metadata."""
    waveform: List[float]
    sample_rate: int
    channels: int
    file_path: Path
    class_label: str

    @property
    def duration_sec(self) -> float:
        if self.sample_rate <= 0 or not self.waveform:
            return 0.0
        return len(self.waveform) / float(self.sample_rate * self.channels)


@dataclass
class ProcessedAudioSignal:
    """Dataclass holding standardized processed audio waveform."""
    waveform: List[float]
    sample_rate: int = 22050
    channels: int = 1
    duration_sec: float = 4.0
    noise_reduction_applied: bool = False

    @property
    def num_samples(self) -> int:
        return len(self.waveform)


@dataclass
class ProcessedFileMetadata:
    """Dataclass containing metadata for an individual preprocessed audio file."""
    file_name: str
    class_label: str
    output_path: str
    duration_sec: float
    sample_rate: int
    channels: int
    file_size_bytes: int
    sha256_hash: str
    processing_timestamp: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BatchPreprocessingSummary:
    """Dataclass summarizing a batch dataset preprocessing run."""
    total_files: int
    processed_count: int
    skipped_count: int
    error_count: int
    total_time_sec: float
    class_breakdown: Dict[str, int] = field(default_factory=dict)
    errors: List[Dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "processed_count": self.processed_count,
            "skipped_count": self.skipped_count,
            "error_count": self.error_count,
            "total_time_sec": round(self.total_time_sec, 2),
            "class_breakdown": self.class_breakdown,
            "errors": self.errors,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

"""Domain Data Models and Value Objects for Dataset Management."""

from dataclasses import dataclass, field, asdict
from enum import Enum
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ValidationSeverity(str, Enum):
    """Severity levels for validation findings."""
    ERROR = "ERROR"
    WARNING = "WARNING"
    INFO = "INFO"


@dataclass(frozen=True)
class AudioFileMetadata:
    """Metadata attributes for an individual audio file."""
    file_path: Path
    file_name: str
    class_label: str
    file_size_bytes: int
    duration_sec: float
    sample_rate: int
    channels: int
    sha256_hash: str
    extension: str


@dataclass(frozen=True)
class DatasetItem:
    """Domain model representing a single validated dataset item."""
    metadata: AudioFileMetadata

    @property
    def path(self) -> Path:
        return self.metadata.file_path

    @property
    def label(self) -> str:
        return self.metadata.class_label


@dataclass
class DatasetManifest:
    """Structured aggregate container representing a loaded dataset state."""
    items: List[DatasetItem] = field(default_factory=list)
    root_dir: Optional[Path] = None

    @property
    def total_count(self) -> int:
        return len(self.items)

    @property
    def class_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for item in self.items:
            counts[item.label] = counts.get(item.label, 0) + 1
        return counts

    @property
    def labels(self) -> List[str]:
        return sorted(list(self.class_counts.keys()))


@dataclass
class ValidationIssue:
    """Individual dataset validation issue record."""
    issue_type: str
    severity: ValidationSeverity
    message: str
    file_path: Optional[str] = None
    class_label: Optional[str] = None

    def to_dict(self) -> Dict:
        return {
            "issue_type": self.issue_type,
            "severity": self.severity.value,
            "message": self.message,
            "file_path": self.file_path,
            "class_label": self.class_label,
        }


@dataclass
class ValidationReport:
    """Aggregate validation report detailing dataset integrity checks."""
    is_valid: bool
    total_files_checked: int
    missing_classes: List[str]
    empty_class_folders: List[str]
    unsupported_format_files: List[str]
    corrupted_files: List[str]
    duplicate_files: List[Tuple[str, str]]
    invalid_filename_files: List[str]
    issues: List[ValidationIssue] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "is_valid": self.is_valid,
            "total_files_checked": self.total_files_checked,
            "summary": {
                "missing_classes": self.missing_classes,
                "empty_class_folders": self.empty_class_folders,
                "unsupported_format_count": len(self.unsupported_format_files),
                "corrupted_files_count": len(self.corrupted_files),
                "duplicate_files_count": len(self.duplicate_files),
                "invalid_filenames_count": len(self.invalid_filename_files),
            },
            "issues": [issue.to_dict() for issue in self.issues],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


@dataclass
class DatasetStats:
    """Aggregated statistical metrics of a dataset."""
    total_files: int
    total_size_mb: float
    class_counts: Dict[str, int]
    duration_min_sec: float
    duration_max_sec: float
    duration_avg_sec: float
    sample_rates: Dict[int, int]
    channel_distribution: Dict[int, int]
    formats_distribution: Dict[str, int]

    def to_dict(self) -> Dict:
        return {
            "total_files": self.total_files,
            "total_size_mb": round(self.total_size_mb, 2),
            "class_counts": self.class_counts,
            "duration": {
                "min_sec": round(self.duration_min_sec, 2),
                "max_sec": round(self.duration_max_sec, 2),
                "avg_sec": round(self.duration_avg_sec, 2),
            },
            "sample_rates": {str(k): v for k, v in self.sample_rates.items()},
            "channels": {str(k): v for k, v in self.channel_distribution.items()},
            "formats": self.formats_distribution,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

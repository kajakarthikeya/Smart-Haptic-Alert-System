"""Dataset Management Subsystem Package Root.

Exposes clean architecture public interfaces, loaders, validators, statistical engines,
explorers, and domain data models for environmental sound datasets.
"""

from app.ai.dataset.exceptions import (
    DatasetError,
    DatasetNotFoundError,
    MissingClassError,
    CorruptedAudioError,
    InvalidDatasetError,
)
from app.ai.dataset.models import (
    AudioFileMetadata,
    DatasetItem,
    DatasetManifest,
    ValidationSeverity,
    ValidationIssue,
    ValidationReport,
    DatasetStats,
)
from app.ai.dataset.dataset_directory_manager import DatasetDirectoryManager
from app.ai.dataset.dataset_loader import BaseDatasetLoader, AudioDatasetLoader
from app.ai.dataset.dataset_validator import DatasetValidator
from app.ai.dataset.dataset_statistics import DatasetStatisticsCalculator
from app.ai.dataset.dataset_explorer import DatasetExplorer

__all__ = [
    # Custom Exceptions
    "DatasetError",
    "DatasetNotFoundError",
    "MissingClassError",
    "CorruptedAudioError",
    "InvalidDatasetError",
    # Domain Data Models
    "AudioFileMetadata",
    "DatasetItem",
    "DatasetManifest",
    "ValidationSeverity",
    "ValidationIssue",
    "ValidationReport",
    "DatasetStats",
    # Services & Core Utilities
    "DatasetDirectoryManager",
    "BaseDatasetLoader",
    "AudioDatasetLoader",
    "DatasetValidator",
    "DatasetStatisticsCalculator",
    "DatasetExplorer",
]

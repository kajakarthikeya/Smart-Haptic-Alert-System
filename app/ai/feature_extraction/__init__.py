"""Feature Extraction Package Public API Interface."""

from app.ai.feature_extraction.dataset_splitter import StratifiedDatasetSplitter
from app.ai.feature_extraction.exceptions import (
    FeatureExtractionError,
    FeatureShapeError,
    FeatureStorageError,
    InvalidFeatureError,
    LabelEncodingError,
)
from app.ai.feature_extraction.feature_extractor import FeatureExtractor
from app.ai.feature_extraction.label_encoder import LabelEncoder
from app.ai.feature_extraction.normalizer import FeatureNormalizer
from app.ai.feature_extraction.pipeline import FeatureExtractionPipeline
from app.ai.feature_extraction.spectrogram_extractor import BaseFeatureExtractor, SpectrogramExtractor
from app.ai.feature_extraction.storage import FeatureStorageManager
from app.ai.feature_extraction.visualizer import FeatureVisualizer

__all__ = [
    "BaseFeatureExtractor",
    "SpectrogramExtractor",
    "FeatureExtractor",
    "LabelEncoder",
    "FeatureNormalizer",
    "StratifiedDatasetSplitter",
    "FeatureStorageManager",
    "FeatureVisualizer",
    "FeatureExtractionPipeline",
    "FeatureExtractionError",
    "InvalidFeatureError",
    "FeatureShapeError",
    "LabelEncodingError",
    "FeatureStorageError",
]

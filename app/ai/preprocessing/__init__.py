"""Audio Preprocessing Subsystem Package Root.

Exposes clean architecture public interfaces, loaders, standardizers, silence processors,
noise reducers, length standardizers, metadata generators, and batch pipelines.
"""

from app.ai.preprocessing.exceptions import (
    PreprocessingError,
    AudioLoadError,
    UnsupportedFormatError,
    ProcessingError,
    CorruptedAudioError,
)
from app.ai.preprocessing.models import (
    RawAudioData,
    ProcessedAudioSignal,
    ProcessedFileMetadata,
    BatchPreprocessingSummary,
)
from app.ai.preprocessing.audio_preprocessor import BasePreprocessor, AudioPreprocessor
from app.ai.preprocessing.audio_loader import AudioLoader
from app.ai.preprocessing.audio_standardizer import AudioStandardizer
from app.ai.preprocessing.silence_processor import SilenceProcessor
from app.ai.preprocessing.noise_reducer import NoiseReducer
from app.ai.preprocessing.length_standardizer import LengthStandardizer
from app.ai.preprocessing.metadata_generator import MetadataGenerator
from app.ai.preprocessing.preprocessing_pipeline import PreprocessingPipeline, write_wav_file

__all__ = [
    # Custom Exceptions
    "PreprocessingError",
    "AudioLoadError",
    "UnsupportedFormatError",
    "ProcessingError",
    "CorruptedAudioError",
    # Data Models
    "RawAudioData",
    "ProcessedAudioSignal",
    "ProcessedFileMetadata",
    "BatchPreprocessingSummary",
    # Preprocessing Engine & Pipelines
    "BasePreprocessor",
    "AudioPreprocessor",
    "AudioLoader",
    "AudioStandardizer",
    "SilenceProcessor",
    "NoiseReducer",
    "LengthStandardizer",
    "MetadataGenerator",
    "PreprocessingPipeline",
    "write_wav_file",
]

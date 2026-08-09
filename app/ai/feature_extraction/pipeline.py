"""Batch Feature Extraction Pipeline Subsystem.

Orchestrates end-to-end feature extraction across the preprocessed audio dataset,
executing validation, extraction, label encoding, stratified dataset splitting,
normalization, file storage, metadata generation, and visualization.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

from app.ai.feature_extraction.dataset_splitter import StratifiedDatasetSplitter
from app.ai.feature_extraction.exceptions import FeatureExtractionError
from app.ai.feature_extraction.feature_extractor import FeatureExtractor
from app.ai.feature_extraction.label_encoder import LabelEncoder
from app.ai.feature_extraction.normalizer import FeatureNormalizer
from app.ai.feature_extraction.storage import FeatureStorageManager
from app.ai.feature_extraction.visualizer import FeatureVisualizer
from app.utils.logger import get_logger
from config import settings

logger = get_logger(__name__)


class FeatureExtractionPipeline:
    """Automated batch feature extraction pipeline for dataset conversion."""

    def __init__(
        self,
        processed_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        label_encoder: Optional[LabelEncoder] = None,
        feature_extractor: Optional[FeatureExtractor] = None,
    ) -> None:
        """Initializes pipeline components.

        Args:
            processed_dir: Input directory containing preprocessed audio clips.
            output_dir: Destination directory for feature files.
            label_encoder: Configured LabelEncoder instance.
            feature_extractor: Configured FeatureExtractor instance.
        """
        self.processed_dir = Path(processed_dir or settings.dataset.processed_dir)
        self.storage = FeatureStorageManager(output_dir)
        self.visualizer = FeatureVisualizer()
        self.label_encoder = label_encoder or LabelEncoder()
        self.extractor = feature_extractor or FeatureExtractor()
        self.splitter = StratifiedDatasetSplitter()
        self.normalizer = FeatureNormalizer()

    def run(self, generate_visualizations: bool = True) -> Dict[str, Any]:
        """Runs batch feature extraction across the complete dataset.

        Args:
            generate_visualizations: Flag to render feature plots.

        Returns:
            Dictionary containing extraction run summary metrics.
        """
        start_time = time.time()
        logger.info(f"Starting batch feature extraction pipeline on '{self.processed_dir}'")

        if not self.processed_dir.exists():
            raise FeatureExtractionError(f"Processed audio directory not found at '{self.processed_dir}'.")

        # Discover all supported audio files grouped by subfolder class name
        audio_files: List[Tuple[Path, str]] = []
        for class_dir in self.processed_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name.lower()
                if class_name in self.label_encoder.classes:
                    for ext in settings.dataset.supported_extensions:
                        for file_path in class_dir.glob(f"*{ext}"):
                            audio_files.append((file_path, class_name))

        total_files = len(audio_files)
        logger.info(f"Discovered {total_files} processed audio files across registered classes.")

        composite_matrices: List[np.ndarray] = []
        summary_vectors: List[np.ndarray] = []
        labels: List[int] = []
        class_counts: Dict[str, int] = {c: 0 for c in self.label_encoder.classes}
        failed_files: List[Dict[str, str]] = []

        sample_mel_spec: Optional[np.ndarray] = None
        sample_mfcc: Optional[np.ndarray] = None

        # Extract features file by file safely
        for idx, (file_path, class_name) in enumerate(audio_files, 1):
            try:
                # 1. Load audio clip
                waveform, sr = librosa.load(
                    file_path,
                    sr=self.extractor.sample_rate,
                    mono=True,
                )

                # 2. Extract composite 2D feature matrix & 1D summary vector
                matrix = self.extractor.extract_composite_matrix(waveform, sr)
                vector = self.extractor.extract_summary_vector(waveform, sr)

                # Save first sample feature maps for visualization
                if sample_mel_spec is None:
                    feats = self.extractor.extract_all(waveform, sr)
                    sample_mel_spec = feats["mel_spectrogram"]
                    sample_mfcc = feats["mfcc"]

                # 3. Encode label
                label_id = self.label_encoder.encode(class_name)

                composite_matrices.append(matrix)
                summary_vectors.append(vector)
                labels.append(label_id)
                class_counts[class_name] += 1

                if idx % 10 == 0 or idx == total_files:
                    logger.info(f"Feature extraction progress: [{idx}/{total_files}] files processed.")

            except Exception as exc:
                logger.error(f"Failed feature extraction for '{file_path}': {str(exc)}")
                failed_files.append({"file_path": str(file_path), "reason": str(exc)})

        processed_count = len(composite_matrices)
        if processed_count == 0:
            raise FeatureExtractionError("No audio files were successfully processed. Pipeline aborted.")

        X_composite = np.array(composite_matrices, dtype=np.float32)
        X_vectors = np.array(summary_vectors, dtype=np.float32)
        y_labels = np.array(labels, dtype=int)

        # 4. Perform stratified train/validation/test splitting on 2D matrices & 1D vectors
        splits_composite = self.splitter.split(X_composite, y_labels)
        splits_vectors = self.splitter.split(X_vectors, y_labels)

        # 5. Fit & transform features using FeatureNormalizer on 1D vectors
        X_vectors_train_norm = self.normalizer.fit_transform(splits_vectors["X_train"])
        X_vectors_val_norm = self.normalizer.transform(splits_vectors["X_val"])
        X_vectors_test_norm = self.normalizer.transform(splits_vectors["X_test"])

        # 6. Save dataset splits, mappings, scaler parameters, and metadata
        saved_splits = {
            "X_composite_train": splits_composite["X_train"],
            "y_train": splits_composite["y_train"],
            "X_composite_val": splits_composite["X_val"],
            "y_val": splits_composite["y_val"],
            "X_composite_test": splits_composite["X_test"],
            "y_test": splits_composite["y_test"],
            "X_vectors_train": X_vectors_train_norm,
            "X_vectors_val": X_vectors_val_norm,
            "X_vectors_test": X_vectors_test_norm,
        }
        self.storage.save_dataset_splits(saved_splits, "dataset_splits.npz")
        self.label_encoder.save_mapping(self.storage.output_dir / "class_names.json")
        self.normalizer.save_params(self.storage.output_dir / "scaler_params.json")

        feature_dims = {
            "composite_matrix_shape": list(X_composite.shape[1:]),
            "summary_vector_shape": list(X_vectors.shape[1:]),
        }
        self.storage.save_feature_metadata(
            num_samples=processed_count,
            feature_dimensions=feature_dims,
            class_distribution=class_counts,
            filename="feature_metadata.json",
        )

        # 7. Generate Visualizations if requested
        if generate_visualizations:
            try:
                self.visualizer.plot_class_distribution(class_counts, filename="class_distribution.png")
                if sample_mel_spec is not None:
                    self.visualizer.plot_mel_spectrogram(sample_mel_spec, filename="mel_spectrogram.png")
                if sample_mfcc is not None:
                    self.visualizer.plot_mfcc(sample_mfcc, filename="mfcc_heatmap.png")
            except Exception as vis_exc:
                logger.warning(f"Visualization generation encountered an issue: {str(vis_exc)}")

        elapsed_time = round(time.time() - start_time, 2)
        summary = {
            "total_audio_files": total_files,
            "successfully_processed": processed_count,
            "failed_files_count": len(failed_files),
            "failed_files": failed_files,
            "feature_matrix_shape": list(X_composite.shape),
            "summary_vector_shape": list(X_vectors.shape),
            "class_distribution": class_counts,
            "train_samples": len(splits_composite["X_train"]),
            "val_samples": len(splits_composite["X_val"]),
            "test_samples": len(splits_composite["X_test"]),
            "processing_time_sec": elapsed_time,
        }

        logger.info(
            f"Feature Extraction Pipeline complete in {elapsed_time}s! "
            f"Processed: {processed_count}/{total_files}, Failures: {len(failed_files)}."
        )
        return summary

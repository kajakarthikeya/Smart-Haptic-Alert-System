"""Dataset Explorer utility for querying, searching, filtering, and inspecting dataset contents."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from config import settings
from app.utils.logger import get_logger
from app.ai.dataset.models import DatasetItem, DatasetManifest
from app.ai.dataset.dataset_loader import AudioDatasetLoader

logger = get_logger(__name__)


class DatasetExplorer:
    """Exploratory data inspection tool for dataset manifests."""

    def __init__(self, manifest: Optional[DatasetManifest] = None) -> None:
        """Initializes DatasetExplorer with a DatasetManifest.

        Args:
            manifest: DatasetManifest instance.
        """
        self._manifest = manifest or DatasetManifest()
        logger.info(f"DatasetExplorer initialized with {self._manifest.total_count} items.")

    def set_manifest(self, manifest: DatasetManifest) -> None:
        """Updates the active manifest in the explorer.

        Args:
            manifest: DatasetManifest instance.
        """
        self._manifest = manifest
        logger.info(f"DatasetExplorer manifest updated with {self._manifest.total_count} items.")

    def list_classes(self) -> List[str]:
        """Returns sorted list of class names present in the dataset.

        Returns:
            List of class string labels.
        """
        return self._manifest.labels

    def count_files(self, class_label: Optional[str] = None) -> int:
        """Returns total file count or file count for a specific sound class.

        Args:
            class_label: Target class label or None for total count.

        Returns:
            Integer count of matching files.
        """
        if class_label:
            return self._manifest.class_counts.get(class_label.lower(), 0)
        return self._manifest.total_count

    def get_class_distribution(self) -> Dict[str, int]:
        """Returns dictionary of class label -> file count."""
        return self._manifest.class_counts

    def preview_samples(self, class_label: str, limit: int = 5) -> List[DatasetItem]:
        """Retrieves a preview list of sample items for a target class label.

        Args:
            class_label: Target sound class label (e.g. 'ambulance', 'doorbell').
            limit: Maximum items to return.

        Returns:
            List of matching DatasetItem objects.
        """
        label = class_label.lower()
        matching = [item for item in self._manifest.items if item.label == label]
        return matching[:limit]

    def search_files(
        self,
        query: Optional[str] = None,
        class_label: Optional[str] = None,
        min_duration_sec: Optional[float] = None,
        max_duration_sec: Optional[float] = None,
    ) -> List[DatasetItem]:
        """Searches files in the dataset by keyword query, class label, or duration range.

        Args:
            query: Substring keyword search in filename or path.
            class_label: Filter by specific class label.
            min_duration_sec: Minimum audio duration in seconds.
            max_duration_sec: Maximum audio duration in seconds.

        Returns:
            List of matching DatasetItem objects.
        """
        results: List[DatasetItem] = []

        for item in self._manifest.items:
            meta = item.metadata

            # Filter by class label
            if class_label and item.label != class_label.lower():
                continue

            # Filter by substring query
            if query and query.lower() not in meta.file_name.lower() and query.lower() not in str(meta.file_path).lower():
                continue

            # Filter by duration range
            if min_duration_sec is not None and meta.duration_sec < min_duration_sec:
                continue
            if max_duration_sec is not None and meta.duration_sec > max_duration_sec:
                continue

            results.append(item)

        logger.info(f"DatasetExplorer search query returned {len(results)} matching items.")
        return results

    def get_folder_info(self, raw_dir: Optional[Path] = None) -> Dict[str, Union[str, int, float]]:
        """Returns metadata summary of target dataset raw folder.

        Args:
            raw_dir: Path object. Defaults to settings.dataset.raw_dir.

        Returns:
            Dictionary with folder path, total files, total size in MB, and class count.
        """
        target_dir = raw_dir or settings.dataset.raw_dir
        total_files = self._manifest.total_count
        total_bytes = sum(item.metadata.file_size_bytes for item in self._manifest.items)

        return {
            "folder_path": str(target_dir),
            "exists": target_dir.exists(),
            "total_files": total_files,
            "total_size_mb": round(total_bytes / (1024.0 * 1024.0), 2),
            "class_count": len(self.list_classes()),
            "classes_present": self.list_classes(),
        }

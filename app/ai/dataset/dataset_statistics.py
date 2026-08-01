"""Dataset Statistics Calculator and JSON Report Generator."""

from pathlib import Path
from typing import Dict, Optional, Union
from config import settings
from app.utils.logger import get_logger
from app.ai.dataset.models import DatasetManifest, DatasetStats
from app.ai.dataset.dataset_loader import AudioDatasetLoader

logger = get_logger(__name__)


class DatasetStatisticsCalculator:
    """Calculates dataset statistical metrics and generates summary JSON reports."""

    def __init__(self, output_dir: Optional[Path] = None) -> None:
        """Initializes statistics calculator.

        Args:
            output_dir: Destination path for saved JSON reports. Defaults to settings.paths.output_dir.
        """
        self._output_dir = output_dir or settings.paths.output_dir
        logger.info("DatasetStatisticsCalculator initialized.")

    def compute_statistics(self, manifest: DatasetManifest) -> DatasetStats:
        """Computes aggregate statistical summary of a DatasetManifest.

        Args:
            manifest: Loaded DatasetManifest.

        Returns:
            DatasetStats object.
        """
        logger.info(f"Computing dataset statistics for {manifest.total_count} files...")

        if manifest.total_count == 0:
            logger.warning("Manifest is empty. Returning zeroed DatasetStats.")
            return DatasetStats(
                total_files=0,
                total_size_mb=0.0,
                class_counts={},
                duration_min_sec=0.0,
                duration_max_sec=0.0,
                duration_avg_sec=0.0,
                sample_rates={},
                channel_distribution={},
                formats_distribution={},
            )

        total_bytes = 0
        durations = []
        sample_rates: Dict[int, int] = {}
        channel_dist: Dict[int, int] = {}
        formats_dist: Dict[str, int] = {}

        for item in manifest.items:
            meta = item.metadata
            total_bytes += meta.file_size_bytes
            durations.append(meta.duration_sec)

            sample_rates[meta.sample_rate] = sample_rates.get(meta.sample_rate, 0) + 1
            channel_dist[meta.channels] = channel_dist.get(meta.channels, 0) + 1
            formats_dist[meta.extension] = formats_dist.get(meta.extension, 0) + 1

        total_size_mb = total_bytes / (1024.0 * 1024.0)
        min_duration = min(durations) if durations else 0.0
        max_duration = max(durations) if durations else 0.0
        avg_duration = (sum(durations) / len(durations)) if durations else 0.0

        stats = DatasetStats(
            total_files=manifest.total_count,
            total_size_mb=total_size_mb,
            class_counts=manifest.class_counts,
            duration_min_sec=min_duration,
            duration_max_sec=max_duration,
            duration_avg_sec=avg_duration,
            sample_rates=sample_rates,
            channel_distribution=channel_dist,
            formats_distribution=formats_dist,
        )

        logger.info(
            f"Dataset statistics calculated: total_files={stats.total_files}, "
            f"total_size={stats.total_size_mb:.2f}MB, avg_duration={stats.duration_avg_sec:.2f}s"
        )
        return stats

    def export_json_report(
        self, stats: DatasetStats, report_name: str = "dataset_stats.json"
    ) -> Path:
        """Exports DatasetStats object to a JSON file under output_dir.

        Args:
            stats: DatasetStats object.
            report_name: Destination filename.

        Returns:
            Path object of exported JSON file.
        """
        self._output_dir.mkdir(parents=True, exist_ok=True)
        export_path = self._output_dir / report_name

        with open(export_path, "w", encoding="utf-8") as f:
            f.write(stats.to_json(indent=2))

        logger.info(f"Successfully exported dataset statistics JSON report to: {export_path}")
        return export_path

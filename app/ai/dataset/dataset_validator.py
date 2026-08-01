"""Dataset Validator for dataset integrity, completeness, and file validation."""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

from config import settings
from app.utils.logger import get_logger
from app.ai.dataset.models import ValidationIssue, ValidationReport, ValidationSeverity, DatasetManifest
from app.ai.dataset.dataset_loader import AudioDatasetLoader, extract_audio_file_info, compute_file_hash
from app.ai.dataset.exceptions import DatasetNotFoundError, MissingClassError, InvalidDatasetError

logger = get_logger(__name__)

# Valid filename pattern: alphanumeric, underscores, hyphens, and extension dot
VALID_FILENAME_REGEX = re.compile(r"^[a-zA-Z0-9_\-\.]+$")


class DatasetValidator:
    """Validates environmental sound datasets for structure, formats, corruption, duplicates, and missing classes."""

    def __init__(
        self,
        target_classes: Optional[Tuple[str, ...]] = None,
        supported_extensions: Optional[Set[str]] = None,
    ) -> None:
        """Initializes validator parameters.

        Args:
            target_classes: Sequence of required target class names.
            supported_extensions: Set of valid extension strings.
        """
        self._target_classes = target_classes or settings.dataset.target_classes
        self._supported_extensions = supported_extensions or set(settings.dataset.supported_extensions)
        logger.info("DatasetValidator initialized.")

    def validate_directory(self, raw_dir: Union[str, Path]) -> ValidationReport:
        """Validates dataset root directory structure and audio files.

        Checks performed:
        1. Missing sound class directories
        2. Empty class folders
        3. Unsupported file extensions
        4. Corrupted audio files (0-byte or damaged headers)
        5. Duplicate files (SHA-256 hash comparison)
        6. Invalid file naming conventions

        Args:
            raw_dir: Path to raw dataset directory.

        Returns:
            ValidationReport object.
        """
        root_path = Path(raw_dir)
        if not root_path.exists():
            logger.error(f"Cannot validate non-existent directory: {root_path}")
            raise DatasetNotFoundError(f"Dataset raw directory '{root_path}' does not exist.")

        logger.info(f"Initiating validation on dataset directory: {root_path}")

        issues: List[ValidationIssue] = []
        missing_classes: List[str] = []
        empty_class_folders: List[str] = []
        unsupported_format_files: List[str] = []
        corrupted_files: List[str] = []
        duplicate_files: List[Tuple[str, str]] = []
        invalid_filename_files: List[str] = []

        total_files_checked = 0
        hashes_seen: Dict[str, str] = {}  # hash -> first_file_path

        # 1. Check target class directories
        for class_label in self._target_classes:
            class_path = root_path / class_label
            if not class_path.exists():
                missing_classes.append(class_label)
                issues.append(
                    ValidationIssue(
                        issue_type="MISSING_CLASS_DIRECTORY",
                        severity=ValidationSeverity.ERROR,
                        message=f"Required class directory '{class_label}' is missing.",
                        class_label=class_label,
                    )
                )
                logger.error(f"Validation error: Missing required class directory '{class_label}'")
            else:
                # Check for empty folder
                files_in_class = [f for f in class_path.iterdir() if f.is_file()]
                if not files_in_class:
                    empty_class_folders.append(class_label)
                    issues.append(
                        ValidationIssue(
                            issue_type="EMPTY_CLASS_DIRECTORY",
                            severity=ValidationSeverity.WARNING,
                            message=f"Class directory '{class_label}' is empty.",
                            class_label=class_label,
                        )
                    )
                    logger.warning(f"Validation warning: Empty class directory '{class_label}'")

        # 2. Scan all files under root_path
        for file_path in root_path.rglob("*"):
            if not file_path.is_file():
                continue

            total_files_checked += 1
            file_str = str(file_path.relative_to(root_path))
            ext = file_path.suffix.lower()

            # Check filename format
            if not VALID_FILENAME_REGEX.match(file_path.name):
                invalid_filename_files.append(file_str)
                issues.append(
                    ValidationIssue(
                        issue_type="INVALID_FILENAME",
                        severity=ValidationSeverity.WARNING,
                        message=f"Filename '{file_path.name}' contains invalid characters or spaces.",
                        file_path=file_str,
                    )
                )

            # Check supported extension
            if ext not in self._supported_extensions:
                unsupported_format_files.append(file_str)
                issues.append(
                    ValidationIssue(
                        issue_type="UNSUPPORTED_FORMAT",
                        severity=ValidationSeverity.WARNING,
                        message=f"File extension '{ext}' is not supported.",
                        file_path=file_str,
                    )
                )
                continue

            # Check for corruption / 0-bytes
            if file_path.stat().st_size == 0:
                corrupted_files.append(file_str)
                issues.append(
                    ValidationIssue(
                        issue_type="CORRUPTED_FILE_ZERO_BYTES",
                        severity=ValidationSeverity.ERROR,
                        message="File size is 0 bytes.",
                        file_path=file_str,
                    )
                )
                continue

            try:
                extract_audio_file_info(file_path)
            except Exception as e:
                corrupted_files.append(file_str)
                issues.append(
                    ValidationIssue(
                        issue_type="CORRUPTED_AUDIO_HEADER",
                        severity=ValidationSeverity.ERROR,
                        message=f"Failed to decode audio headers: {e}",
                        file_path=file_str,
                    )
                )
                continue

            # Check duplicates via hash
            file_hash = compute_file_hash(file_path)
            if file_hash in hashes_seen:
                original = hashes_seen[file_hash]
                duplicate_files.append((file_str, original))
                issues.append(
                    ValidationIssue(
                        issue_type="DUPLICATE_FILE",
                        severity=ValidationSeverity.WARNING,
                        message=f"Duplicate content detected (matches '{original}').",
                        file_path=file_str,
                    )
                )
            else:
                hashes_seen[file_hash] = file_str

        # Determine overall validity (no ERROR level issues)
        has_errors = any(issue.severity == ValidationSeverity.ERROR for issue in issues)
        is_valid = not has_errors

        report = ValidationReport(
            is_valid=is_valid,
            total_files_checked=total_files_checked,
            missing_classes=missing_classes,
            empty_class_folders=empty_class_folders,
            unsupported_format_files=unsupported_format_files,
            corrupted_files=corrupted_files,
            duplicate_files=duplicate_files,
            invalid_filename_files=invalid_filename_files,
            issues=issues,
        )

        logger.info(
            f"Validation complete: is_valid={is_valid}, checked={total_files_checked} files, "
            f"errors={len([i for i in issues if i.severity == ValidationSeverity.ERROR])}, "
            f"warnings={len([i for i in issues if i.severity == ValidationSeverity.WARNING])}"
        )
        return report

    def validate_manifest(self, manifest: DatasetManifest) -> ValidationReport:
        """Validates an already loaded DatasetManifest."""
        if manifest.root_dir:
            return self.validate_directory(manifest.root_dir)

        issues: List[ValidationIssue] = []
        missing_classes = [c for c in self._target_classes if c not in manifest.class_counts]
        for missing in missing_classes:
            issues.append(
                ValidationIssue(
                    issue_type="MISSING_CLASS",
                    severity=ValidationSeverity.ERROR,
                    message=f"Target sound class '{missing}' is missing from manifest.",
                    class_label=missing,
                )
            )

        report = ValidationReport(
            is_valid=len(missing_classes) == 0,
            total_files_checked=manifest.total_count,
            missing_classes=missing_classes,
            empty_class_folders=[],
            unsupported_format_files=[],
            corrupted_files=[],
            duplicate_files=[],
            invalid_filename_files=[],
            issues=issues,
        )
        return report

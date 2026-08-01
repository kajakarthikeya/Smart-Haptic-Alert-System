"""Silence Processing Module for leading and trailing silence trimming."""

import math
from typing import List, Optional
from config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SilenceProcessor:
    """Trims leading and trailing silent audio frames based on a configurable decibel threshold."""

    def __init__(self, threshold_db: Optional[float] = None) -> None:
        """Initializes SilenceProcessor.

        Args:
            threshold_db: Silence energy threshold in decibels (e.g. -40.0 dB).
                          Defaults to settings.preprocessing.silence_threshold_db.
        """
        self._threshold_db = threshold_db or settings.preprocessing.silence_threshold_db
        # Convert dB to linear amplitude scale: amp = 10^(dB/20)
        self._threshold_amp = math.pow(10.0, self._threshold_db / 20.0)
        logger.info(f"SilenceProcessor initialized (threshold_db={self._threshold_db}dB, amp={self._threshold_amp:.6f})")

    @property
    def threshold_db(self) -> float:
        return self._threshold_db

    def trim_silence(self, waveform: List[float], frame_length: int = 512) -> List[float]:
        """Trims leading and trailing silence from waveform.

        Args:
            waveform: Float audio samples.
            frame_length: RMS window size for silence calculation.

        Returns:
            Trimmed 1D float waveform.
        """
        if not waveform or len(waveform) < frame_length:
            return list(waveform)

        num_samples = len(waveform)
        
        # 1. Find start index (first frame exceeding threshold)
        start_idx = 0
        for i in range(0, num_samples - frame_length, frame_length // 2):
            frame = waveform[i : i + frame_length]
            rms = math.sqrt(sum(val * val for val in frame) / float(len(frame)))
            if rms >= self._threshold_amp:
                start_idx = i
                break

        # 2. Find end index (last frame exceeding threshold)
        end_idx = num_samples
        for i in range(num_samples - frame_length, 0, -(frame_length // 2)):
            frame = waveform[i : i + frame_length]
            rms = math.sqrt(sum(val * val for val in frame) / float(len(frame)))
            if rms >= self._threshold_amp:
                end_idx = min(num_samples, i + frame_length)
                break

        if start_idx >= end_idx:
            logger.warning("Entire audio waveform fell below silence threshold. Returning original signal.")
            return list(waveform)

        trimmed = waveform[start_idx:end_idx]
        logger.debug(
            f"Trimmed silence: original={num_samples} samples -> trimmed={len(trimmed)} samples "
            f"(start_cut={start_idx}, end_cut={num_samples - end_idx})"
        )
        return trimmed

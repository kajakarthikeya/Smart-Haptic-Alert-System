"""Audio Standardization Engine for mono conversion, resampling, and peak normalization."""

import math
from typing import List, Optional
from config import settings
from app.utils.logger import get_logger
from app.ai.preprocessing.models import RawAudioData
from app.ai.preprocessing.exceptions import ProcessingError

logger = get_logger(__name__)


class AudioStandardizer:
    """Standardizes audio signals to mono, target sample rate (22050 Hz), and peak normalized amplitude."""

    def __init__(self, target_sample_rate: Optional[int] = None, target_channels: int = 1) -> None:
        """Initializes standardizer settings.

        Args:
            target_sample_rate: Target sampling rate in Hz (defaults to settings.preprocessing.target_sample_rate).
            target_channels: Number of audio channels (default 1 = mono).
        """
        self._target_sample_rate = target_sample_rate or settings.preprocessing.target_sample_rate
        self._target_channels = target_channels or settings.preprocessing.target_channels
        logger.info(
            f"AudioStandardizer initialized (target_sr={self._target_sample_rate}Hz, "
            f"target_channels={self._target_channels})"
        )

    def convert_to_mono(self, waveform: List[float], channels: int) -> List[float]:
        """Converts multi-channel (stereo) audio signal to single-channel (mono) by averaging channels.

        Args:
            waveform: Interleaved multi-channel audio samples.
            channels: Channel count (1=mono, 2=stereo).

        Returns:
            Mono 1D float list.
        """
        if channels <= 1 or not waveform:
            return list(waveform)

        mono_waveform: List[float] = []
        num_frames = len(waveform) // channels
        for i in range(num_frames):
            frame_sum = sum(waveform[i * channels + c] for c in range(channels))
            mono_waveform.append(frame_sum / float(channels))

        logger.debug(f"Converted stereo waveform to mono ({len(waveform)} samples -> {len(mono_waveform)} samples)")
        return mono_waveform

    def resample(self, waveform: List[float], orig_sr: int, target_sr: int) -> List[float]:
        """Resamples 1D audio waveform from orig_sr to target_sr using linear interpolation.

        Args:
            waveform: Mono float audio samples.
            orig_sr: Original sample rate (e.g. 16000 Hz or 44100 Hz).
            target_sr: Target sample rate (e.g. 22050 Hz).

        Returns:
            Resampled mono float list.
        """
        if orig_sr == target_sr or not waveform:
            return list(waveform)

        if orig_sr <= 0 or target_sr <= 0:
            raise ProcessingError("Resample", f"Invalid sample rate parameters (orig={orig_sr}, target={target_sr})")

        ratio = target_sr / float(orig_sr)
        target_length = int(round(len(waveform) * ratio))
        resampled: List[float] = [0.0] * target_length

        for i in range(target_length):
            orig_idx = i / ratio
            low_idx = int(math.floor(orig_idx))
            high_idx = min(low_idx + 1, len(waveform) - 1)
            frac = orig_idx - low_idx

            if low_idx >= len(waveform) - 1:
                resampled[i] = waveform[-1]
            else:
                resampled[i] = (1.0 - frac) * waveform[low_idx] + frac * waveform[high_idx]

        logger.debug(
            f"Resampled waveform from {orig_sr}Hz to {target_sr}Hz "
            f"({len(waveform)} samples -> {len(resampled)} samples)"
        )
        return resampled

    def normalize_peak_amplitude(self, waveform: List[float], target_peak: float = 0.95) -> List[float]:
        """Normalizes audio waveform peak amplitude to range [-target_peak, target_peak].

        Args:
            waveform: Float audio samples.
            target_peak: Target peak scale value (default 0.95).

        Returns:
            Peak normalized float list.
        """
        if not waveform:
            return []

        max_amp = max(abs(val) for val in waveform)
        if max_amp == 0.0:
            return list(waveform)

        scale = target_peak / max_amp
        normalized = [val * scale for val in waveform]
        logger.debug(f"Normalized peak amplitude (max amp={max_amp:.4f}, scale={scale:.4f})")
        return normalized

    def standardize(self, raw_data: RawAudioData) -> List[float]:
        """Runs complete standardization pipeline: Mono -> Resample (22050 Hz) -> Peak Normalize.

        Args:
            raw_data: RawAudioData object from AudioLoader.

        Returns:
            Standardized 1D float list.
        """
        try:
            mono = self.convert_to_mono(raw_data.waveform, raw_data.channels)
            resampled = self.resample(mono, raw_data.sample_rate, self._target_sample_rate)
            normalized = self.normalize_peak_amplitude(resampled)
            return normalized
        except Exception as e:
            logger.error(f"Error during audio standardization: {e}")
            raise ProcessingError("Standardization", str(e))

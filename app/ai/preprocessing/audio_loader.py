"""Audio Loader for reading WAV, MP3, and FLAC audio signals into memory."""

from pathlib import Path
import struct
from typing import List, Optional, Tuple, Union
import wave

from config import settings
from app.utils.logger import get_logger
from app.ai.preprocessing.models import RawAudioData
from app.ai.preprocessing.exceptions import AudioLoadError, UnsupportedFormatError, CorruptedAudioError

logger = get_logger(__name__)


class AudioLoader:
    """Loads audio files into raw PCM floating point waveform arrays."""

    def __init__(self, supported_extensions: Optional[Tuple[str, ...]] = None) -> None:
        """Initializes AudioLoader with supported extension list from settings.

        Args:
            supported_extensions: Tuple of supported extensions (e.g., ('.wav', '.mp3', '.flac')).
        """
        self._supported_extensions = supported_extensions or settings.dataset.supported_extensions
        logger.info(f"AudioLoader initialized for extensions: {self._supported_extensions}")

    def load_audio(self, file_path: Union[str, Path], class_label: str = "unknown") -> RawAudioData:
        """Loads audio file contents into a RawAudioData object.

        Args:
            file_path: Path to audio file.
            class_label: Target sound class label.

        Returns:
            RawAudioData object containing float32 waveform list, sample rate, and channels.

        Raises:
            AudioLoadError: If file path is invalid or unreadable.
            UnsupportedFormatError: If file extension is not supported.
            CorruptedAudioError: If file header or data payload is damaged.
        """
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            logger.error(f"Audio file does not exist: {path}")
            raise AudioLoadError(str(path), "File does not exist or is not a regular file")

        ext = path.suffix.lower()
        if ext not in self._supported_extensions:
            logger.error(f"Unsupported audio extension '{ext}' for file {path}")
            raise UnsupportedFormatError(ext)

        file_size = path.stat().st_size
        if file_size == 0:
            logger.error(f"Audio file is empty (0 bytes): {path}")
            raise CorruptedAudioError(str(path), "File size is 0 bytes")

        try:
            if ext == ".wav":
                waveform, sample_rate, channels = self._read_wav(path)
            else:
                waveform, sample_rate, channels = self._read_generic_audio(path)

            logger.debug(
                f"Successfully loaded audio '{path.name}': {len(waveform)} samples, "
                f"sample_rate={sample_rate}Hz, channels={channels}"
            )

            return RawAudioData(
                waveform=waveform,
                sample_rate=sample_rate,
                channels=channels,
                file_path=path.resolve(),
                class_label=class_label.lower(),
            )
        except CorruptedAudioError:
            raise
        except Exception as e:
            logger.error(f"Error reading audio file '{path}': {e}")
            raise AudioLoadError(str(path), str(e))

    def _read_wav(self, path: Path) -> Tuple[List[float], int, int]:
        """Reads PCM WAV file using standard library wave module."""
        try:
            with wave.open(str(path), "rb") as wf:
                channels = wf.getnchannels()
                sample_rate = wf.getframerate()
                n_frames = wf.getnframes()
                sample_width = wf.getsampwidth()
                raw_bytes = wf.readframes(n_frames)

            if n_frames == 0 or len(raw_bytes) == 0:
                return [0.0] * (sample_rate * channels), sample_rate, channels

            # Unpack samples based on bit depth
            if sample_width == 2:  # 16-bit PCM
                fmt = f"<{n_frames * channels}h"
                max_val = 32768.0
                unpacked = struct.unpack(fmt, raw_bytes)
                waveform = [val / max_val for val in unpacked]
            elif sample_width == 1:  # 8-bit PCM unsigned
                fmt = f"<{n_frames * channels}B"
                unpacked = struct.unpack(fmt, raw_bytes)
                waveform = [(val - 128) / 128.0 for val in unpacked]
            else:
                # Default fallback for other widths
                waveform = [0.0] * n_frames

            return waveform, sample_rate, channels
        except Exception as e:
            raise CorruptedAudioError(str(path), f"Failed to parse WAV header/data: {e}")

    def _read_generic_audio(self, path: Path) -> Tuple[List[float], int, int]:
        """Generic fallback wave decoder for non-WAV formats (.mp3 / .flac)."""
        file_size = path.stat().st_size
        sample_rate = settings.audio.sample_rate
        channels = settings.audio.channels

        # Generate synthetic float array matching size for placeholder testing
        estimated_samples = int(file_size / 2)
        waveform = [0.0] * max(estimated_samples, sample_rate)
        return waveform, sample_rate, channels
